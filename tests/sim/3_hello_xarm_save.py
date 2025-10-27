import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import time
import numpy as np
from lerobot.envs.factory import make_env, make_env_config
from pathlib import Path
from lerobot.utils.io_utils import write_video  # 导入视频保存函数
import pprint


videos_dir = Path("./videos")
videos_dir.mkdir(parents=True, exist_ok=True)

env_cfg = make_env_config("xarm")
env = make_env(env_cfg, n_envs=10, use_async_envs=False)
print(env.metadata)
observation, info = env.reset()

frames = []

for i in range(1000):
    if isinstance(env, gym.vector.SyncVectorEnv):
        frame = env.envs[0].render()
    else:
        frame = env.call("render")[0]
    
    frames.append(frame)

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    done = np.logical_or(terminated, truncated)
    if np.any(done):
        if len(frames) > 0:
            video_path = videos_dir / f"xarm_episode_{len(os.listdir(videos_dir))}.mp4"
            # 获取渲染帧率，如果无法获取则默认为30
            fps = env.unwrapped.metadata.get("render_fps", 30)
            write_video(str(video_path), np.array(frames), fps)
            print(f"保存视频到: {video_path}")

            frames = []
        
        observation, info = env.reset_wait()
        print("重置环境")
    
    time.sleep(0.01)

# 如果还有未保存的帧，保存最后一个视频
if len(frames) > 0:
    video_path = videos_dir / f"xarm_episode_{len(os.listdir(videos_dir))}.mp4"
    fps = env.unwrapped.metadata.get("render_fps", 30)
    write_video(str(video_path), np.array(frames), fps)
    print(f"保存最终视频到: {video_path}")

env.close()