#%% Setup
import torch as t
from nnsight import LanguageModel
from utils import MemoryMonitor, import_modules_and_load_models, get_weights
from huggingface_hub import list_repo_files
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.express as px
from nnsight import LanguageModel
import einops
import gc
import math
import os
import seaborn as sns
import pandas as pd

monitor, models = import_modules_and_load_models()

# %% Plot the diff 

lm = models["llama-3.1-8b"]
lm_ft = models["llama-3.1-8b-r1-distilled"]

emb, emb_ft = get_weights(
    lm, lm_ft, "embed_tokens", None
)

#%% Compare embeddings 

import torch.nn.functional as F
from tqdm import tqdm
def compare_embeddings(
    lm_base_name: str,
    lm_ft_name: str,
    metric: str = "cossim",
    plot_hist: bool = True,
    save_plot: bool = False,
    save_path: str = "output",
    plot_name: str = "",
):
    emb, emb_ft = get_weights(
        models[lm_base_name], models[lm_ft_name], "embed_tokens", None
    )
    # shape: (n_tokens, d_model)
    if metric == "diff_norm":
        diff = emb - emb_ft
        diff_norm = t.norm(diff, p="fro", dim=1)
        emb_norm = t.norm(emb, p="fro", dim=1)
        emb_ft_norm = t.norm(emb_ft, p="fro", dim=1)
        rel_diff = diff_norm / (emb_norm + emb_ft_norm)
    elif metric == "cossim":
        # use the HS inner product to measure the similarity between matrices
        rel_diff = F.cosine_similarity(emb, emb_ft, dim=1)
    
    if plot_hist:
        fig = px.histogram(
            x=rel_diff.to(t.float32).cpu().numpy(),
            nbins=50,
            title=f"{lm_base_name} and {lm_ft_name} by {metric} difference at embedding",
            labels={'x': f"{metric} difference", 'y': 'Frequency'}
        )
        fig.update_layout(yaxis_type="log")
        fig.show()
        
        if save_plot:
            fig.write_image(f"{save_path}/diff_{metric}_{plot_name}.png")
    
    return rel_diff


batch_processing_models = [
    # ("llama-3.1-8b", "llama-3.1-8b-r1-distilled", "mlp_llama_base_distilled"),
    ("qwen-2.5-7b", "qwen-2.5-7b-math", "emb_qwen_base_math"),
    ("qwen-2.5-7b", "qwen-2.5-7b-instruct", "emb_qwen_base_inst"),
    ("qwen-2.5-7b-math", "qwen-2.5-7b-math-instruct", "emb_qwen_math_mathinst"),
]

for lm_base_name, lm_ft_name, plot_name in tqdm(batch_processing_models):
    cossim = compare_embeddings(
        lm_base_name=lm_base_name,
        lm_ft_name=lm_ft_name,
        metric="cossim",
        plot_hist=True,
        save_plot=True,
        plot_name=plot_name,
        save_path="output/cossim"
    )
    

#%% Testing for relative embedding difference

def compare_relative_embedding_difference(
    lm_base_name: str,
    lm_ft_name: str,
    metric: str = "cossim",
    plot_hist: bool = True,
    save_plot: bool = False,
    save_path: str = "output/rel_embed",
    plot_name: str = "",
):
    if save_plot:
        os.makedirs(save_path, exist_ok=True)
    
    emb, emb_ft = get_weights(
        models[lm_base_name], models[lm_ft_name], "embed_tokens", None
    )
    
    
    
    



        