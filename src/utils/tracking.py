"""
WandB and HuggingFace integration helpers.
Handles configuration robustness and graceful failures.
"""
import wandb
import os
import warnings

def init_wandb(config, job_type="train"):
    """
    Initialize WandB run with robust error handling.
    
    Args:
        config (dict): Configuration dictionary.
        job_type (str): Type of job (train, eval, etc).
        
    Returns:
        wandb.run or None: The wandb run object if successful, else None.
    """
    # Check if WANDB is disabled via env var or config
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        print("WandB disabled via WANDB_MODE environment variable.")
        return None

    # Handle missing entity/project gracefully
    entity = config.get('wandb_entity')
    if entity == "your-entity" or not entity:
        warnings.warn("WandB entity not set or default placeholder used. Logging might be restricted or local-only.")
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
        print("Continuing execution without logging...")
        return None