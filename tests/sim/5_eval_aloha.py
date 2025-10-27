import os
from pathlib import Path
import gym_aloha
import gymnasium as gym
import torch
import time
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.io_utils import write_video  # 导入视频保存函数

device = "cuda"

videos_dir = Path("./videos")
videos_dir.mkdir(parents=True, exist_ok=True)

# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
# pretrained_policy_path = "lerobot/diffusion_pusht"
pretrained_policy_path = Path("outputs/train/aloha_transfer_cube")

# ACTPolicy_cfg = make_policy_config("act")

policy = ACTPolicy.from_pretrained(pretrained_policy_path)

env = gym.make(
    "gym_aloha/AlohaTransferCube-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
    render_mode='rgb_array',
)

policy.reset()
numpy_observation, info = env.reset(seed=4)

step = 0
done = False
frames = []
while not done:
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"]["top"])

    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    observation = {
        "observation.state": state,
        "observation.images.top": image,
    }

    with torch.inference_mode():
        action = policy.select_action(observation)

    numpy_action = action.squeeze(0).to("cpu").numpy()

    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    frame = env.render()
    frames.append(frame)
    done = terminated | truncated | done
    if done:
        if len(frames) > 0:
            video_path = f"aloha_episode_{len(os.listdir(videos_dir))}.mp4"
            fps = env.unwrapped.metadata.get("render_fps", 30)
            write_video(str(video_path), np.array(frames), fps)
            print(f"保存视频到: {video_path}")
            frames = []
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")


