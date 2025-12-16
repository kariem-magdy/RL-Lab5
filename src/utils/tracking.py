"""
WandB and HuggingFace integration helpers.
"""
import wandb
import os

def init_wandb(config, job_type="train"):
    """Initialize WandB run."""
    run = wandb.init(
        project=config['wandb_project'],
        entity=config['wandb_entity'],
        config=config,
        job_type=job_type,
        mode="online" if os.environ.get("WANDB_API_KEY") else "disabled"
    )
    return run