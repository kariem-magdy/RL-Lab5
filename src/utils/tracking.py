"""
WandB and HuggingFace integration helpers.
Handles configuration robustness and graceful failures.
"""
import wandb
import os
import warnings

def init_wandb(config, job_type="train"):
    # Check if WANDB is disabled via env var
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        print("WandB disabled via WANDB_MODE environment variable.")
        return None

    # Handle missing entity/project gracefully
    entity = config.get('wandb_entity')
    if entity == "your-entity" or not entity:
        entity = None # Let wandb use user's default

    project = config.get('wandb_project', 'world-models-rl')

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            job_type=job_type,
            mode=os.environ.get("WANDB_MODE", "online"),
            reinit=True
        )
        return run
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")
        return None