#%%
import os 
import sys
os.chdir("/workspace/sae-diff/src")
import torch as t
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm.auto import tqdm
from sae_modules import GatedDiffSAE, GatedSAEConfig
from data_prep import new_cached_activation_generator as activation_generator
from nnsight import LanguageModel
from transformers import AutoTokenizer
from datasets import load_dataset
from memory_util import print_gpu_memory, MemoryMonitor
from transformers import get_constant_schedule_with_warmup
import wandb

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config



# monitor = MemoryMonitor()
# monitor.start()
# monitor.start_continuous_monitoring()

def train(cfg: Dict[str, Any], use_monitor: bool = False):
    
    ### 0. --- Setup ---
    t.manual_seed(cfg["seed"])
    if use_monitor:
        monitor = MemoryMonitor()
        monitor.start()
        # monitor.start_continuous_monitoring()
    
    ### 1. --- Setup data generator and load models ---
    base_model = LanguageModel(
        cfg["model_base"]["path"],
        device_map=cfg["device"],
        torch_dtype=t.bfloat16
    )
    finetune_model = LanguageModel(
        cfg["model_ft"]["path"],
        device_map=cfg["device"],
        torch_dtype=t.bfloat16
    )
    base_tokenizer = AutoTokenizer.from_pretrained(cfg["model_base"]["path"])
    finetune_tokenizer = AutoTokenizer.from_pretrained(cfg["model_ft"]["path"])
    
    print("Models loaded")
    
    if use_monitor:
        # print_gpu_memory()
        monitor.measure("Loaded models")


    # generator config
    gcfg = cfg["generator"]
    dataset = load_dataset(
        gcfg["dataset"]["name"],
        'v1',
        split=gcfg["dataset"]["split"],
        streaming=gcfg["dataset"]["streaming"]
    )
    dataset = dataset.shuffle(seed=cfg["seed"])
    
    train_data_generator = activation_generator(
        base_model=base_model,
        finetune_model=finetune_model,
        base_tokenizer=base_tokenizer,
        finetune_tokenizer=finetune_tokenizer,
        dataset=dataset,
        layer_num=gcfg["layer_num"],
        activation_batch_size=gcfg["activation_batch_size"],
        generator_batch_size=gcfg["generator_batch_size"],
        acts_per_run=gcfg["acts_per_run"],
        tokens_per_example=gcfg["tokens_per_example"],
        skip_first_n_tokens=gcfg["skip_first_n_tokens"],
        device=cfg["device"]
    )
    # TODO: add checks and warnings for incompatible parameters
    
    ### 2. --- Setup SAE and optimizers ---
    # Initialize SAE
    sae_config = GatedSAEConfig(
        sparsity_coeff=cfg["sae"]["sparsity_coeff"],
        aux_coeff=cfg["sae"]["aux_coeff"],
        d_model=cfg["sae"]["d_model"],
        d_sae=cfg["sae"]["d_sae"],
        weight_normalize_eps=cfg["sae"]["weight_normalize_eps"],
        standardize_method=cfg["sae"]["standardize_method"]
    )
    sae = GatedDiffSAE(sae_config, cfg["device"])
    expansion_ratio = cfg["sae"]["d_sae"] / cfg["sae"]["d_model"]
    
    ocfg = cfg["optimizer"]
    if ocfg["name"] == "adamw":
        optimizer = AdamW(
            params=sae.parameters(),
            lr=float(ocfg["lr"]),
            weight_decay=float(ocfg["weight_decay"]),
            betas=(float(ocfg["beta1"]), float(ocfg["beta2"]))
        )
    else:
        raise NotImplementedError(f"Optimizer {ocfg['name']} not implemented")
    
    scfg = cfg["scheduler"]
    if scfg["name"] == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(scfg["warmup_steps"])
        )
    else:
        raise NotImplementedError(f"Scheduler {scfg['name']} not implemented")

    print("Scheduler and optimizer setup complete")
    ### 3. --- Setup logger ---
    wcfg = cfg["wandb"]
    if wcfg["use_wandb"]:
        run = wandb.init(
            project=wcfg["project"],
            name=wcfg["name"] # dependent on the config file name
        )
    
    ### 4. --- Training setup ---
    tcfg = cfg["trainer"]
    virtual_batch_size = int(tcfg["virtual_batch_size"])
    actual_batch_size = int(tcfg["actual_batch_size"])
    tokens_per_example = int(tcfg["tokens_per_example"])
    skip_first_n_tokens = int(tcfg["skip_first_n_tokens"])
    
    # Verify accumulation_steps matches the prescribed value in tcfg
    accumulation_steps = virtual_batch_size // actual_batch_size
    prescribed_accumulation_steps = int(tcfg["accumulation_steps"])
    if accumulation_steps != prescribed_accumulation_steps:
        print(f"WARNING: Computed accumulation_steps ({accumulation_steps}) does not match prescribed value in config ({prescribed_accumulation_steps})")
        print(f"Using computed value value: {prescribed_accumulation_steps}")

    num_optimizer_steps = int(tcfg["num_optimizer_steps"])
    prescribed_num_optimizer_steps = int(tcfg["num_optimizer_steps"])
    if num_optimizer_steps != prescribed_num_optimizer_steps:
        print(f"WARNING: Computed num_optimizer_steps ({num_optimizer_steps}) does not match prescribed value in config ({prescribed_num_optimizer_steps})")
        print(f"Using computed value value: {prescribed_num_optimizer_steps}")
    
    total_forward_passes = int(tcfg["total_forward_passes"])
    prescribed_total_forward_passes = int(tcfg["total_forward_passes"])
    if total_forward_passes != prescribed_total_forward_passes:
        print(f"WARNING: Computed total_forward_passes ({total_forward_passes}) does not match prescribed value in config ({prescribed_total_forward_passes})")
        print(f"Using computed value value: {prescribed_total_forward_passes}")
    
    # alpha = tcfg["sparsity_loss_alpha"] already taken care of in the forward pass of the SAE
    pbar = tqdm(range(num_optimizer_steps))
    
    print("Training setup complete")
    monitor.measure("Training setup complete")
    
    ### 5. --- Training loop ---
    
    # The active latent lists 
    frac_active_list = []
    
    for optimizer_step in pbar:
        optimizer.zero_grad()
        if use_monitor:
            if (optimizer_step+1) % 64 == 0:
                monitor.measure(label= f"Step {optimizer_step+1}", print_msg=True)
            else:
                monitor.measure(label= f"Step {optimizer_step+1}", print_msg=False)
        
        # accumulate the gradients to perform a single optimizer step
        for acc_step in range(accumulation_steps):
            # Get a new batch of data
            try:
                # monitor.measure("Generating batch")
                batch_data = next(train_data_generator).to(cfg["device"]).to(t.float32)
                # monitor.measure("Batch generated")
            except TypeError as e:
                print(f"Invalid data encountered: {e}")
                print("Attempting to load a different batch...")
                while True:
                    try:
                        batch_data = next(train_data_generator).to(cfg["device"]).to(t.float32)
                        break
                    except TypeError as e:
                        print(f"Invalid data encountered: {e}")
                        print("Attempting to load a different batch...")
            
            # Now successfully generated batch_data with shape (batch_size, 2*d_model)
            # which we can then use directly in the forward pass
            
            # Get pre-activations for dead latents only
            loss_dict, loss, acts_post, diff_stdized, diff_reconstructed, scale_cache = sae.forward(batch_data)
            
            # Keep track of fraction of active latents over the batch
            # average over the batch dimension
            # to be used for resampling
            frac_active = (acts_post.abs() > 1e-8).float().mean(0)
            # frac_active is of shape (d_sae,)
            frac_active_list.append(frac_active)
            
            # accumulate the gradients!
            # print(loss.shape) -> result: [64] == batch size
            loss.mean().backward()
            
            if acc_step == 0:
                # at each optimizer step, log the result
                # display the progress bar
                
                with t.inference_mode():
                    if wcfg["use_wandb"]:
                        # calculate the variance explained
                        # In evaluation model we need to compare the prediction to the original data instead of the standardized ones
                        if cfg["sae"]["standardize_method"] == "plain":
                            fvu = (diff_stdized - diff_reconstructed).pow(2).mean() / (diff_stdized.pow(2).mean() + float(cfg["sae"]["weight_normalize_eps"]))
                        elif cfg["sae"]["standardize_method"] == "per_token":
                            diff_recon_unscale = diff_reconstructed * (scale_cache["std"] + float(cfg["sae"]["weight_normalize_eps"])) + scale_cache["mu"]
                            diff_unscale = diff_stdized * (scale_cache["std"] + float(cfg["sae"]["weight_normalize_eps"])) + scale_cache["mu"]
                            fvu = (diff_unscale - diff_recon_unscale).pow(2).mean() / ((diff_unscale - diff_unscale.mean(0)).pow(2).mean() + float(cfg["sae"]["weight_normalize_eps"]))
                        elif cfg["sae"]["standardize_method"] == "per_batch":
                            diff_recon_unscale = diff_reconstructed * (scale_cache["norm_scale"] + float(cfg["sae"]["weight_normalize_eps"])) + scale_cache["mu"]
                            diff_unscale = diff_stdized * (scale_cache["norm_scale"] + float(cfg["sae"]["weight_normalize_eps"])) + scale_cache["mu"]
                            fvu = (diff_unscale - diff_recon_unscale).pow(2).mean() / ((diff_unscale - diff_unscale.mean(0)).pow(2).mean() + float(cfg["sae"]["weight_normalize_eps"]))

                        # store some scalars for logging
                        run_data = {
                            "recon_loss": loss_dict["L_reconstruction"].mean(0).sum().item(),
                            "aux_loss": loss_dict["L_aux"].mean(0).sum().item(),
                            "sparsity_loss": loss_dict["L_sparsity"].mean(0).sum().item(),
                            "loss": loss.mean(0).sum().item(),
                            "fvu": fvu,
                            "dead_latents": frac_active.sum().item(),
                            "memory_usage": monitor.measurements[-1][1]
                        }
                    
                    pbar.set_postfix(
                        lr=scheduler.get_last_lr()[0],
                        loss=loss.mean(0).sum().item(),
                        frac_active=frac_active.mean().item(),
                        **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                        fvu=fvu
                    )
                    run.log(run_data)
                    
        optimizer.step()
        scheduler.step()

        if optimizer_step % cfg["save_checkpoint_freq"] == 0 or optimizer_step == num_optimizer_steps-1:
            os.makedirs(cfg["save_dir"], exist_ok=True)
            sae_path = os.path.join(cfg["save_dir"], f"sae_checkpoint_{optimizer_step}.pt")
            t.save(sae.state_dict(), sae_path)

        # Implement resampling after each time a threshold is crossed
        if (optimizer_step + 1) % cfg["resample"]["freq"] == 0:
            # This would release the memory by a bit?
            frac_active_list = frac_active_list[-cfg["resample"]["window"]:]
            frac_active_in_window = t.stack(frac_active_list, dim=0)
            # reset the weights via resampling
            # just use the last batch data as the input, they only serve to determine a baseline distribution for weight re-activations anyway
            sae.resampling(batch_data, frac_active_in_window, cfg["resample"]["scale"])
            if use_monitor:
                monitor.measure(f"After resampling {optimizer_step}")
        

    run.finish()
    if use_monitor:
        monitor.report()
        return monitor
            

# if __name__ == "__main__":
# monitor = MemoryMonitor()
# monitor.start()
# monitor.start_continuous_monitoring()
cfg = load_config("/workspace/sae-diff/configs/gated_sae_toy.yaml")
monitor = train(cfg, use_monitor=True)

#%% 

# Analyze memory monitor
import matplotlib.pyplot as plt
monitor.plot()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    