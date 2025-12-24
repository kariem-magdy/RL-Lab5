import argparse
import yaml
import torch
import os
import sys
from huggingface_hub import HfApi, upload_file

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_lstm import MDNLSTM
from src.models.controller import Controller

def main(args):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in environment variables.")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'])
    print(f"🚀 Publishing to Hugging Face Hub... Model: {args.repo_name}")

    api = HfApi()
    user = api.whoami(token=token)['name']
    repo_id = f"{user}/{args.repo_name}"
    
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # 1. Upload Config
    upload_file(
        path_or_fileobj=args.config,
        path_in_repo="config.yaml",
        repo_id=repo_id,
        repo_type="model"
    )

    # 2. Upload Checkpoints
    files_to_upload = {
        "vae_best.pth": "vae.pth",
        "lstm_best.pth": "lstm.pth",
        "controller_dream_best.pth": "controller.pth"
    }

    for local_name, remote_name in files_to_upload.items():
        path = os.path.join(config['checkpoint_dir'], local_name)
        if os.path.exists(path):
            print(f"Uploading {local_name}...")
            upload_file(
                path_or_fileobj=path,
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model"
            )
        else:
            print(f"⚠️ Warning: {local_name} not found, skipping.")

    print(f"✅ Upload Complete! View at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--repo_name", default="world-models-breakout")
    args = parser.parse_args()
    main(args)