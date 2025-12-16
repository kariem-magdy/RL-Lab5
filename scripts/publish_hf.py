import argparse
from huggingface_hub import HfApi, create_repo
import os

def main(args):
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        return

    print(f"Creating repo {args.repo_id}...")
    try:
        create_repo(args.repo_id, repo_type="model", token=token)
    except Exception as e:
        print(f"Repo might exist: {e}")

    print("Uploading artifacts...")
    # Upload checkpoints
    api.upload_folder(
        folder_path="results/checkpoints",
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo="checkpoints",
        token=token
    )
    # Upload Config
    api.upload_file(
        path_or_fileobj="configs/default.yaml",
        path_in_repo="config.yaml",
        repo_id=args.repo_id,
        token=token
    )
    # Upload Results
    api.upload_folder(
        folder_path="results/eval_runs",
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo="results",
        token=token
    )
    
    print("Upload Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    main(args)