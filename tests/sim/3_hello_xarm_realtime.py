import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import time
import cv2
import numpy as np
from lerobot.envs.factory import make_env, make_env_config

env_cfg = make_env_config("xarm")
env = make_env(env_cfg, n_envs=10, use_async_envs=False)

observation, info = env.reset()

# 创建可视化窗口
cv2.namedWindow("XArm Environment", cv2.WINDOW_NORMAL)
cv2.resizeWindow("XArm Environment", 800, 600)

for i in range(1000):
    # 渲染当前状态
    if isinstance(env, gym.vector.SyncVectorEnv):
        frame = env.envs[0].render()
    else:
        frame = env.call("render")[0]
    
    # 转换为 BGR 格式供 OpenCV 使用
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("XArm Environment", frame)
        
        # 处理窗口事件，按 ESC 键退出
        key = cv2.waitKey(1)
        if key == 27:  # ESC 键
            break
    
    # 采样随机动作
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    done = np.logical_or(terminated, truncated)
    if np.any(done):
        observation, info = env.reset()
        print("重置环境")
    
    # 添加小延迟使可视化更流畅
    time.sleep(0.01)

env.close()
cv2.destroyAllWindows()
