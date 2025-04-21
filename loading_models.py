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

#%%
import os
# os.environ['HF_HOME'] = "/tmp"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")