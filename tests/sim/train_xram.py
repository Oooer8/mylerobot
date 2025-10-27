# example.py
import os
os.environ["MUJOCO_GL"] = "glfw"
import gymnasium as gym
import gym_xarm

env = gym.make("gym_xarm/XarmLift-v0", 
               obs_type="pixels_agent_pos",
               render_mode="human")
observation, info = env.reset()

while True:
    action = env.action_space.sample()
    print(action)
    
    observation, reward, terminated, truncated, info = env.step(action)
    print("obs after step", observation["agent_pos"])
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()