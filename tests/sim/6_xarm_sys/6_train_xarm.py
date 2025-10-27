from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import get_policy_class, make_policy_config, make_policy


def train():
    output_directory = Path("outputs/train/xarm_lift")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(output_directory / "logs")

    device = torch.device("cuda")

    # Number of offline training steps
    training_steps = 5000
    log_freq = 100
    save_freq = 1000  # Frequency for model checkpoints
    max_grad_norm = 1.0

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="oooer/xarm-lift",
        root="./datasets"
        )

    ACT_policy_cfg = make_policy_config("act",
                                        chunk_size = 100, 
                                        n_action_steps = 50, )
    
    policy = make_policy(ds_meta=dataset_metadata, cfg=ACT_policy_cfg)
    policy.train()

    if ACT_policy_cfg.observation_delta_indices is not None:
        delta_timestamps = {
            "observation.images.top": [i / dataset_metadata.fps for i in ACT_policy_cfg.observation_delta_indices],
            "observation.state": [i / dataset_metadata.fps for i in ACT_policy_cfg.observation_delta_indices],
            "action": [i / dataset_metadata.fps for i in ACT_policy_cfg.action_delta_indices],
        }
    else:
        delta_timestamps = {
            "action": [i / dataset_metadata.fps for i in ACT_policy_cfg.action_delta_indices],
        }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    full_dataset = LeRobotDataset(
        repo_id="oooer/xarm-lift",
        root="./datasets",
        delta_timestamps=delta_timestamps
        )

    train_dataloader = torch.utils.data.DataLoader(
        full_dataset,
        num_workers=4,
        batch_size=16,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps)

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in train_dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Log periodically
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f} lr: {current_lr:.6f}")
                # Log training metrics to TensorBoard
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("LearningRate", current_lr, step)
                
                # Log individual loss components if available
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f"Loss/{loss_name}", loss_value, step)
            
            # Save checkpoint periodically
            if step % save_freq == 0 and step > 0:
                checkpoint_dir = output_directory / f"checkpoint_{step}"
                policy.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint at step {step}")
            
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save final model
    policy.save_pretrained(output_directory)
    print(f"Training completed. Final model saved to {output_directory}")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train()