from diffusers import FluxTransformer2DModel
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights
from diffusers.models.model_loading_utils import load_model_dict_into_meta
import safetensors.torch
import glob
import torch

# Initialize model with empty weights
with init_empty_weights():
    config = FluxTransformer2DModel.load_config("black-forest-labs/FLUX.1-dev", subfolder="transformer")
    model = FluxTransformer2DModel.from_config(config)

# Download checkpoints
dev_ckpt = snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", allow_patterns="transformer/*")
krea_ckpt = snapshot_download(repo_id="black-forest-labs/FLUX.1-Krea-dev", allow_patterns="transformer/*")

# Get sorted shard paths
dev_shards = sorted(glob.glob(f"{dev_ckpt}/transformer/*.safetensors"))
krea_shards = sorted(glob.glob(f"{krea_ckpt}/transformer/*.safetensors"))

# Initialize dictionaries for merged and guidance weights
merged_state_dict = {}
guidance_state_dict = {}

# Merge shards
for dev_shard, krea_shard in zip(dev_shards, krea_shards):
    state_dict_dev = safetensors.torch.load_file(dev_shard)
    state_dict_krea = safetensors.torch.load_file(krea_shard)

    # Process keys from dev model
    for k in list(state_dict_dev.keys()):
        if "guidance" in k:
            # Keep guidance weights from dev model
            guidance_state_dict[k] = state_dict_dev.pop(k)
        else:
            # Average non-guidance weights if key exists in krea
            if k in state_dict_krea:
                merged_state_dict[k] = (state_dict_dev.pop(k) + state_dict_krea.pop(k)) / 2
            else:
                raise ValueError(f"Key {k} missing in krea shard.")

    # Check for residual keys in krea (e.g., extra guidance keys)
    for k in list(state_dict_krea.keys()):
        if "guidance" in k:
            # Skip extra guidance keys in krea
            state_dict_krea.pop(k)
        else:
            raise ValueError(f"Unexpected non-guidance key in krea shard: {k}")

    # Verify no unexpected residue
    if len(state_dict_dev) > 0:
        raise ValueError(f"Residue in dev shard: {list(state_dict_dev.keys())}")
    if len(state_dict_krea) > 0:
        raise ValueError(f"Residue in krea shard: {list(state_dict_krea.keys())}")

# Combine merged and guidance state dictionaries
merged_state_dict.update(guidance_state_dict)

# Load merged state dictionary into model
load_model_dict_into_meta(model, merged_state_dict)

# Convert to bfloat16 and save
model.to(torch.bfloat16).save_pretrained("merged/transformer")
