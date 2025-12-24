import wandb
import os

def init_wandb(config, job_type="train"):
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        print("WandB disabled via WANDB_MODE environment variable.")
        return None

    entity = config.get('wandb_entity')
    if entity == "your-entity" or not entity:
        entity = None 

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