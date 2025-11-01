import os
from pathlib import Path
import gym_xarm
import gymnasium as gym
import torch
import time
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.io_utils import write_video  # 导入视频保存函数

device = "cuda"

videos_dir = Path("./videos")
videos_dir.mkdir(parents=True, exist_ok=True)

name = "xarm_lift_moving_keyboard"
# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
# pretrained_policy_path = "lerobot/diffusion_pusht"
pretrained_policy_path = Path(f"outputs/train/{name}")

# ACTPolicy_cfg = make_policy_config("act")

policy = ACTPolicy.from_pretrained(pretrained_policy_path)

env = gym.make(
    "gym_xarm/XarmLiftMoving-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=500,
    render_mode="rgb_array",
    moving_mode="linear",
)

print("Observation Space:", env.observation_space)

camera_names = []
if isinstance(env.observation_space.spaces.get("pixels"), gym.spaces.Dict):
    camera_names = list(env.observation_space.spaces["pixels"].spaces.keys())
    print(f"检测到相机: {camera_names}")

output_directory = Path(f"outputs/videos/{name}")
output_directory.mkdir(parents=True, exist_ok=True)

sucess_episodes = 0
max_episodes = 100
for episode in range(max_episodes):
    policy.reset()
    numpy_observation, info = env.reset(seed=episode)

    step = 0
    done = False

    frames={}
    for camera_name in camera_names:
        frames.update({camera_name:[]})

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
        print(f"{step=} {reward=} {terminated=}")

        frame = env.render()
        for camera_name in camera_names:
            frames[camera_name].append(frame[camera_name])
        
        done = terminated | truncated | done
        if done:
            for frames_key, frames_value in frames.items():
                if len(frames_value) > 0:
                    video_path = f"{output_directory}/xarm_episode_{episode}_{frames_key}.mp4"
                    fps = env.unwrapped.metadata.get("render_fps", 25)
                    write_video(str(video_path), np.array(frames_value), fps)
                    print(f"保存视频到: {video_path}")

                    frames={}
                    for camera_name in camera_names:
                        frames.update({camera_name:[]})
        step += 1

    if terminated:
        sucess_episodes += 1
        print("Success!")
    else:
        print("Failure!")

print(f"成功率: {sucess_episodes / max_episodes * 100}%")

