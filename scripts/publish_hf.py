from huggingface_hub import HfApi, create_repo
import shutil

def main():
    api = HfApi()
    repo_id = "YOUR_USERNAME/world-models-breakout-v4"
    
    try:
        create_repo(repo_id, repo_type="model")
    except:
        pass
        
    print(f"Uploading models to {repo_id}...")
    api.upload_file(path_or_fileobj="vae.pth", path_in_repo="vae.pth", repo_id=repo_id)
    api.upload_file(path_or_fileobj="rnn.pth", path_in_repo="rnn.pth", repo_id=repo_id)
    api.upload_file(path_or_fileobj="controller.pth", path_in_repo="controller.pth", repo_id=repo_id)
    api.upload_file(path_or_fileobj="configs/default.yaml", path_in_repo="config.yaml", repo_id=repo_id)
    
    print("Upload complete!")

if __name__ == "__main__":
    main()