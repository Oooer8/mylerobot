import os
os.environ["MUJOCO_GL"] = "glfw"
import gymnasium as gym
import gym_xarm  # 导入包会执行__init__.py中的注册代码
import time

# 创建环境，明确指定render_mode
env = gym.make("gym_xarm/XarmLift-v0", render_mode="human")

# 获取环境的unwrapped版本，这通常可以访问到更多内部参数
unwrapped_env = env.unwrapped
print("\n环境对象的属性和方法:")
env_attrs = [attr for attr in dir(unwrapped_env) if not attr.startswith('_')]
for attr in env_attrs:
    try:
        value = getattr(unwrapped_env, attr)
        if not callable(value):
            print(f"  - {attr}: {value}")
    except:
        print(f"  - {attr}: 无法获取值")

observation, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: reward = {reward:.3f}")

    image = env.render()

    time.sleep(unwrapped_env.dt)
    
    if terminated or truncated:
        observation, info = env.reset()
        print("环境重置")

env.close()


