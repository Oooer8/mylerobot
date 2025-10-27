from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.envs.configs import HILEnvConfig

# 创建配置
cfg = HILEnvConfig(
    device="cuda",
    fps=30,
    wrapper={
        "use_gripper": True,
        "control_mode": "keyboard_ee",  # 或 "leader", "keyboard_ee"
        "control_time_s": 60,
        "gripper_penalty": -0.1
    }
)

# 确保wrapper是一个对象而不是字典
class WrapperConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# 将字典转换为对象
cfg.wrapper = WrapperConfig(cfg.wrapper)

# 创建环境
env = make_robot_env(cfg)

# 使用环境
obs, info = env.reset()

# 由于没有定义policy，我们需要手动控制或创建一个简单的随机策略
# 这里使用一个简单的随机动作策略作为示例
for _ in range(1000):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break