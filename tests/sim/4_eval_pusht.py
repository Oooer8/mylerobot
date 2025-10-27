from pathlib import Path
import gym_pusht
import gymnasium as gym
import torch
import time
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

device = "cuda"

# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
pretrained_policy_path = "lerobot/diffusion_pusht"
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
    render_mode='human',
)

policy.reset()
numpy_observation, info = env.reset(seed=42)

step = 0
done = False
while not done:
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])

    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    with torch.inference_mode():
        action = policy.select_action(observation)

    numpy_action = action.squeeze(0).to("cpu").numpy()

    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    env.render()
    time.sleep(0.05)

    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")


