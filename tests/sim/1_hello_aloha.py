import gymnasium as gym
import gym_aloha  # 导入包会执行__init__.py中的注册代码

# 创建环境
env = gym.make("gym_aloha/AlohaInsertion-v0")  # 或 "gym_aloha/AlohaTransferCube-v0"

# 使用环境
observation, info = env.reset()
for i in range(1000):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: reward = {reward:.3f}")
    if terminated or truncated:
        observation, info = env.reset()
env.close()