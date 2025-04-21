#%% Set Hugging Face cache to a location with more space
import os
import shutil

# Use the current workspace directory
workspace_dir = os.getcwd()
new_cache_dir = os.path.join(workspace_dir, "huggingface_cache")

# Remove old cache if it exists
if os.path.exists(new_cache_dir):
    shutil.rmtree(new_cache_dir)

# Create new cache directory
os.makedirs(new_cache_dir, exist_ok=True)

# Set environment variables for both transformers and huggingface_hub
os.environ['HF_HOME'] = new_cache_dir
os.environ['TRANSFORMERS_CACHE'] = new_cache_dir

print(f"Using cache directory: {new_cache_dir}")

#%% Setup
import torch as t
import nnsight
from utils import MemoryMonitor
from huggingface_hub import list_repo_files

#%% memory usage
monitor = MemoryMonitor("Diffing Weights")
monitor.start()
monitor.start_continuous_monitoring()

#%% Loading models

model_names = {
    "llama-3.1-8b-r1-distilled": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "qwen-2.5-7b-math-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-7b-math": "Qwen/Qwen2.5-Math-7B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
}



from nnsight import LanguageModel
#%%
models = {}
# Load model with reduced precision (bfloat16)
models["llama-3.1-8b"] = LanguageModel(
    model_names["llama-3.1-8b"], 
    device_map="cuda", 
    dispatch=True,
    torch_dtype=t.bfloat16  # Use bfloat16 precision
)

#%%
# Try loading the distilled model with error handling
# Check available files for the distilled model
from huggingface_hub import list_repo_files
print("Checking available files for DeepSeek model:")
files = list_repo_files(model_names["llama-3.1-8b-r1-distilled"])
print(files)
try:
    models["llama-3.1-8b-r1-distilled"] = LanguageModel(
        model_names["llama-3.1-8b-r1-distilled"], 
        device_map="cuda", 
        dispatch=True,
        torch_dtype=t.bfloat16
    )
except Exception as e:
    print(f"Error loading distilled model: {e}")
    print("This model might require special access or have a different file structure.")

# %%

models["qwen-2.5-7b"] = LanguageModel(
    model_names["qwen-2.5-7b"], 
    device_map="cuda", 
    dispatch=True,
    torch_dtype=t.bfloat16,
    trust_remote_code=True  # Qwen models often require this
)

#%%
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
