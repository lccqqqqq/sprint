# Implementation of Gated SAE
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, Generator
from nnsight import LanguageModel
from jaxtyping import Float
from dataclasses import dataclass
from torch.utils.data import IterableDataset
from torch.distributions import Categorical

@dataclass
class GatedSAEConfig:
    sparsity_coeff: float = 0.01
    aux_coeff: float = 0.01
    d_model: int = 768
    d_sae: int = 1024
    
    # other bookkeeping
    weight_normalize_eps: float = 1e-6
    standardize_method: str = "per_token"
    # accepting "plain", "per_token", "per_batch"
    


class GatedDiffSAE(nn.Module):
    def __init__(self, cfg: GatedSAEConfig, device: t.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        # initialize the parameters
        # By default did not include options for weight-tying
        self.W_dec = nn.Parameter(t.empty(cfg.d_sae, cfg.d_model))
        self.b_dec = nn.Parameter(t.zeros(cfg.d_model))
        self.W_gate = nn.Parameter(t.empty(cfg.d_model, cfg.d_sae))
        self.b_gate = nn.Parameter(t.zeros(cfg.d_sae))
        self.r_mag = nn.Parameter(t.zeros(cfg.d_sae))
        self.b_mag = nn.Parameter(t.zeros(cfg.d_sae))
        
        self._init_weights()
        self.to(self.device)
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_gate)
        nn.init.kaiming_uniform_(self.W_dec)
        # the biases are already initialized to zero

    @property
    def W_mag(self) -> Float[Tensor, "d_model d_sae"]:
        return self.r_mag.exp().unsqueeze(0) * self.W_gate
    

    def forward(
        self,
        x: Float[Tensor, "batch 2d_model"],
    ) -> tuple[
        dict[str, Float[Tensor, "batch"]],
        Float[Tensor, "batch"],
        Float[Tensor, "batch d_sae"],
        Float[Tensor, "batch d_model"],
        Float[Tensor, "batch d_model"],
    ]:
        """
        Implement the forward pass with the gated SAE.
        """
        src = x[:, :self.cfg.d_model] # the pass only acts on the source
        tgt = x[:, self.cfg.d_model:]
        diff = tgt - src
        
        # implement post processing
        if self.cfg.standardize_method == "plain":
            # no normalization/centering
            diff_stdized = diff - self.b_dec
            scale_cache = None
        elif self.cfg.standardize_method == "per_token":
            mu = diff.mean(0)
            std = diff.std(0)
            diff_centered = (diff - mu) / (std + self.cfg.weight_normalize_eps)
            diff_stdized = diff_centered - self.b_dec
            scale_cache = {
                "mu": mu,
                "std": std,
            }
        elif self.cfg.standardize_method == "per_batch":
            mu = diff.mean(0)
            diff_centered_batch = diff - mu
            norm_scale = diff_centered_batch.norm(dim=1).mean()
            diff_centered = diff_centered_batch / (norm_scale + self.cfg.weight_normalize_eps)
            diff_stdized = diff_centered - self.b_dec
            scale_cache = {
                "mu": mu,
                "norm_scale": norm_scale,
            }
        else:
            raise NotImplementedError(f"Invalid standardization method: {self.cfg.standardize_method}")
        

        # Compute the gating terms (pi_gate(x) and f_gate(x) in the paper)
        gating_pre_activation = (
            einops.einsum(diff_stdized, self.W_gate, "batch d_in, d_in d_sae -> batch d_sae") + self.b_gate
        )
        active_features = (gating_pre_activation > 0).float()

        # Compute the magnitude term (f_mag(x) in the paper)
        magnitude_pre_activation = (
            einops.einsum(diff_stdized, self.W_mag, "batch d_in, d_in d_sae -> batch d_sae") + self.b_mag
        )
        feature_magnitudes = F.relu(magnitude_pre_activation)

        # Compute the hidden activations (f˜(x) in the paper)
        acts_post = active_features * feature_magnitudes

        # Compute reconstructed input
        diff_reconstructed = (
            einops.einsum(acts_post, self.W_dec, "batch d_sae, d_sae d_in -> batch d_in") + self.b_dec
        )

        # Compute loss terms
        gating_post_activation = F.relu(gating_pre_activation)
        via_gate_reconstruction = (
            einops.einsum(
                gating_post_activation, self.W_dec.detach(), "batch d_sae, d_sae d_in -> batch d_in"
            )
            + self.b_dec.detach()
        )
        
        
        loss_dict = {
            "L_reconstruction": (diff_reconstructed - diff_stdized).pow(2).mean(-1),
            "L_sparsity": gating_post_activation.sum(-1),
            "L_aux": (via_gate_reconstruction - diff_stdized).pow(2).sum(-1),
        }

        loss = loss_dict["L_reconstruction"] + self.cfg.sparsity_coeff * loss_dict["L_sparsity"] + self.cfg.aux_coeff * loss_dict["L_aux"]

        assert sorted(loss_dict.keys()) == ["L_aux", "L_reconstruction", "L_sparsity"]
        return loss_dict, loss, acts_post, diff_stdized, diff_reconstructed, scale_cache
    
    @t.no_grad()
    def resampling(
        self,
        batch_data: Float[Tensor, "batch 2d_model"], 
        frac_active_in_window: Float[Tensor, "window d_sae"], # this intends to be in the model keeping track of a mask signalling whether a feature is active
        resample_scale: float,
    ) -> None:
        # generate next batch of data with post processing
        # Directly use the batch data without generating new ones. Should be equivalent and more memory efficient...
        # batch_data = next(resampling_generator) # (batch, d_model * 2)
        loss_dict, _, _, diff, _, _ = self.forward(batch_data)
        l2_loss = loss_dict["L_reconstruction"]
        # fraction of active features in the window
        is_dead = (frac_active_in_window < 1e-8).all(dim=0)
        dead_latents = t.nonzero(is_dead).squeeze(-1)
        n_dead = dead_latents.numel()
        if n_dead == 0:
            return  # If we have no dead features, then we don't need to resample

        if l2_loss.max() < 1e-6:
            return

        # Draw `d_sae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
        distn = Categorical(probs=l2_loss.pow(2) / l2_loss.pow(2).sum())
        replacement_indices = distn.sample((n_dead,))  # type: ignore

        # Index into the batch of hidden activations to get our replacement values
        replacement_values = (diff - self.b_dec)[replacement_indices]   # [n_dead d_in]
        replacement_values_normalized = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )
        
        # --- replace decoder weights for dead latents ---
        self.W_dec.data[dead_latents, :] = replacement_values_normalized
        
        # --- replace value-path encoder weights for dead latents ---
        val_norm_alive_mean = self.W_mag[:, ~is_dead].norm(dim=0).mean().item() if (~is_dead).any() else 1.0
        self.W_mag.data[:, dead_latents] = replacement_values_normalized.T * val_norm_alive_mean * resample_scale

        # --- replace gated-path encoder weights for dead latents ---
        gate_norm_alive_mean = self.W_gate[:, ~is_dead].norm(dim=0).mean().item() if (~is_dead).any() else 1.0
        self.W_gate.data[:, dead_latents] = replacement_values_normalized.T * gate_norm_alive_mean * resample_scale
        
        # --- replace the biases for dead latents ---
        self.b_gate.data[dead_latents] = -1.0
        self.b_mag.data[dead_latents] = 0.0
        

        
if __name__ == "__main__":
    # testing out different functionalities 
    cfg = GatedSAEConfig()
    gated_sae = GatedDiffSAE(cfg, device=t.device("cuda"))
    x = t.randn(10, 768*2).to(gated_sae.device)
    loss_dict, loss, acts_post, diff, diff_reconstructed = gated_sae(x)
    print(loss_dict)
    print(loss)
    print(acts_post.shape)
    print(diff_reconstructed.shape)

    gated_sae.resampling(x, t.zeros(1024, 1024), 0.5)
        
        





