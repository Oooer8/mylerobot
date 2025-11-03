from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import gym_xarm
import numpy as np
import time

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import get_policy_class, make_policy_config, make_policy
from lerobot.utils.io_utils import write_video


def evaluate_policy(policy, num_episodes=10, seed_offset=10000):
    """评估策略的成功率"""
    device = torch.device("cuda")
    
    env = gym.make(
        "gym_xarm/XarmLiftMoving-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=500,
        render_mode="rgb_array",
        moving_mode="static",
    )
    
    camera_names = []
    if isinstance(env.observation_space.spaces.get("pixels"), gym.spaces.Dict):
        camera_names = list(env.observation_space.spaces["pixels"].spaces.keys())
    
    success_episodes = 0
    
    for episode in range(num_episodes):
        policy.reset()
        numpy_observation, info = env.reset(seed=episode+seed_offset)
        
        step = 0
        done = False
        
        while not done:
            state = torch.from_numpy(numpy_observation["agent_pos"])
            state = state.to(torch.float32)
            state = state.to(device, non_blocking=True)
            state = state.unsqueeze(0)
            
            processed_images = {}
            for camera_name in camera_names:
                image = torch.from_numpy(numpy_observation["pixels"][camera_name])
                image = image.to(torch.float32) / 255
                image = image.permute(2, 0, 1)  # HWC -> CHW
                image = image.to(device, non_blocking=True)
                image = image.unsqueeze(0)
                processed_images[f"observation.images.{camera_name}"] = image
            
            observation = {"observation.state": state}
            observation.update(processed_images)
            
            with torch.inference_mode():
                action = policy.select_action(observation)
            
            numpy_action = action.squeeze(0).to("cpu").numpy()
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            
            done = terminated | truncated
            step += 1
        
        if terminated:
            success_episodes += 1
    
    success_rate = success_episodes / num_episodes * 100
    env.close()
    return success_rate


def train():
    name = "xarm_lift_moving_keyboard_static_3camera_fixedstart"
    output_directory = Path(f"outputs/train/{name}")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # 创建专门的目录来保存最后的模型和最佳模型
    last_model_dir = output_directory / "last"
    best_model_dir = output_directory / "best"
    last_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(output_directory / "logs")

    device = torch.device("cuda")

    # Number of offline training steps
    training_steps = 5000
    log_freq = 100
    save_freq = 1000  # Frequency for model checkpoints
    eval_freq = 500   # Frequency for evaluation
    max_grad_norm = 1.0
    eval_episodes = 50  # Number of episodes for each evaluation

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id=f"oooer/{name}",
        root=f"./outputs/datasets/{name}"
        )

    ACT_policy_cfg = make_policy_config("act",
                                        chunk_size = 100, 
                                        n_action_steps = 50, )
    
    policy = make_policy(ds_meta=dataset_metadata, cfg=ACT_policy_cfg)
    policy.train()

    # Initialize delta_timestamps dictionary
    delta_timestamps = {}
    dataset_features = dataset_to_policy_features(dataset_metadata.features)
    print("dataset_features", dataset_features)
    for key in dataset_features:
        if "action" in key and ACT_policy_cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in ACT_policy_cfg.action_delta_indices]
        elif "observation" in key and ACT_policy_cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in ACT_policy_cfg.observation_delta_indices]
    
    # We can then instantiate the dataset with these delta_timestamps configuration.
    full_dataset = LeRobotDataset(
        repo_id=f"oooer/{name}",
        root=f"./outputs/datasets/{name}",
        delta_timestamps=delta_timestamps
        )

    train_dataloader = torch.utils.data.DataLoader(
        full_dataset,
        num_workers=8,
        batch_size=128,
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
    best_success_rate = 0.0
    
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
            
            # Save checkpoint and evaluate periodically
            if step % eval_freq == 0 and step > 0:
                # Switch to evaluation mode
                policy.eval()
                
                # Save checkpoint
                checkpoint_dir = output_directory / f"checkpoint_{step}"
                policy.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint at step {step}")
                
                policy.save_pretrained(last_model_dir)
                print(f"Saved as last model at step {step}")
                
                # Evaluate the model
                print(f"Evaluating model at step {step}...")
                success_rate = evaluate_policy(policy, num_episodes=eval_episodes)
                print(f"Step {step} - Success Rate: {success_rate:.2f}%")
                
                # Log success rate to TensorBoard
                writer.add_scalar("Evaluation/SuccessRate", success_rate, step)
                
                # Save best model based on success rate
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    policy.save_pretrained(best_model_dir)
                    print(f"New best model saved with success rate: {best_success_rate:.2f}%")
                    writer.add_scalar("Evaluation/BestSuccessRate", best_success_rate, step)
                
                # Switch back to training mode
                policy.train()
            
            # Regular checkpoint saving
            if step % save_freq == 0 and step > 0 and step % eval_freq != 0:
                checkpoint_dir = output_directory / f"checkpoint_{step}"
                policy.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint at step {step}")
                
                policy.save_pretrained(last_model_dir)
                print(f"Saved as last model at step {step}")
            
            step += 1
            if step >= training_steps:
                done = True
                break

    # Final evaluation
    policy.eval()
    final_success_rate = evaluate_policy(policy, num_episodes=eval_episodes)
    print(f"Final Success Rate: {final_success_rate:.2f}%")
    writer.add_scalar("Evaluation/FinalSuccessRate", final_success_rate, step)
    
    policy.save_pretrained(last_model_dir)
    
    if final_success_rate >= best_success_rate:
        best_success_rate = final_success_rate
        policy.save_pretrained(best_model_dir)
        print(f"Final model saved as new best model with success rate: {best_success_rate:.2f}%")
        writer.add_scalar("Evaluation/BestSuccessRate", best_success_rate, step)
    
    print(f"Training completed. Final model saved to {last_model_dir}")
    print(f"Best model saved to {best_model_dir} with success rate: {best_success_rate:.2f}%")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train()