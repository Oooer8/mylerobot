import os
os.environ["MUJOCO_GL"] = "glfw"
import logging
import time
from collections import deque
from collections.abc import Sequence
from threading import Lock
from typing import Annotated, Any

import gymnasium as gym
import gym_xarm  # 导入包会执行__init__.py中的注册代码
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from pprint import pprint

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop,KeyboardEndEffectorTeleopConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
logging.basicConfig(level=logging.INFO)


class KeySimEnv(gym.Env):
    """
    键盘控制仿真环境

    """

    def __init__(
        self,
        use_gripper: bool = True,
        display_cameras: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.display_cameras = display_cameras
        self.use_gripper = use_gripper
        self.moving_mode = kwargs.get("moving_mode", "linear")

        # teleop keyboard
        self.keyboard_config = KeyboardEndEffectorTeleopConfig(use_gripper=True)
        self.teleop = KeyboardEndEffectorTeleop(self.keyboard_config)
        self.teleop.connect()


        # sim env
        self.sim_env = gym.make(
            "gym_xarm/XarmLiftMoving-v0",
            obs_type="pixels_agent_pos",
            max_episode_steps=500,
            render_mode="rgb_array",
            moving_mode=self.moving_mode,
        )

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None
        self.current_observation = None
        self._get_observation_from_sim()

        self._setup_spaces()

        # # 访问 MuJoCo 模型并提取关节名称
        # joint_names = []
        # for i in range(self.sim_env.model.njnt):
        #     # 获取第 i 个关节的名称
        #     joint_name = self.sim_env.model.joint(i).name
        #     joint_names.append(joint_name)
        # print("🤖所有关节名称：", joint_names)
        # self._joint_names = ["{name}.pos" for name in joint_names if name.startswith("joint")]
        # print("🦾机械臂关节名称：", self._joint_names)
        # # self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        camera_names = []
        for i in range(self.sim_env.model.ncam):
            camera_name = self.sim_env.model.camera(i).name
            camera_names.append(camera_name)
        self._image_keys = [name for name in camera_names if name.startswith("camera")]
        print("📷摄像机名称：", self._image_keys)

        self.last_gripper_value=0

    def _get_observation_from_sim(self) -> dict[str, np.ndarray]:
        obs_dict = self.sim_env.get_obs()
        # print("obs_dict", obs_dict)
        self.current_observation = obs_dict

        # try:
        #     gripper_joint_pos = self.sim_env.unwrapped._utils.get_joint_qpos(
        #         self.sim_env.unwrapped.model, 
        #         self.sim_env.unwrapped.data, 
        #         "right_outer_knuckle_joint"
        #     )
        #     print("夹爪关节位置:", gripper_joint_pos)
        # except:
        #     print("找不到指定关节")
        # joint_positions = np.array([
        #     self.sim_env.unwrapped._utils.get_joint_qpos(
        #         self.sim_env.unwrapped.model, 
        #         self.sim_env.unwrapped.data, 
        #         name
        #     ) for name in self._joint_names])
        # print("夹爪关节位置:", joint_positions)
        # images = {key: obs_dict[key] for key in self._image_keys}
        # self.current_observation = {"agent_pos": joint_positions, "pixels": images}

    def _setup_spaces(self):
        observation_spaces = {}

        # Define observation spaces for images and other states.
        if "pixels" in self.current_observation:
            prefix = "observation.images"
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=self.current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in self.current_observation["pixels"]
            }

        observation_spaces["observation.state"] = gym.spaces.Box(
            low=0,
            high=10,
            shape=self.current_observation["agent_pos"].shape,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            # ctrl_range = self.sim_env.model.actuator_ctrlrange["drive_joint"]
            # drive_joint_lower_bound = ctrl_range[0]
            # drive_joint_upper_bound = ctrl_range[1]
            
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [-1]])
            bounds["max"] = np.concatenate([bounds["max"], [1]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.sim_env.reset()

        # Reset episode tracking variables.
        self.last_gripper_value = 0
        self.current_step = 0
        self.episode_data = None
        self.current_observation = None
        self._get_observation_from_sim()

        return self.current_observation, {"is_intervention": True}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is processed and sent to the robot as joint position commands
        that may be either absolute values or deltas based on the environment configuration.

        Args:
            action: The commanded joint positions as a numpy array or torch tensor.

        Returns:
            A tuple containing:
                - observation (dict): The new sensor observation after taking the step.
                - reward (float): The step reward (default is 0.0 within this wrapper).
                - terminated (bool): True if the episode has reached a terminal state.
                - truncated (bool): True if the episode was truncated (e.g., time constraints).
                - info (dict): Additional debugging information including intervention status.
        """

        action_numpy = self._key_val_mapping(action)
        # print("action", action_numpy)
        action_numpy = np.clip(
            action_numpy, 
            self.sim_env.action_space.low, 
            self.sim_env.action_space.high
        )/5

        observation, reward, terminated, truncated, info = self.sim_env.step(action_numpy)
        
        self.current_observation = observation

        self.current_step += 1

        return (
            self.current_observation,
            reward,
            terminated,
            truncated,
            {"is_intervention": True,
             "reward_components": info["reward_components"]
             },
        )

    def render(self):
        vis_image = self.sim_env.render()
        import cv2
        if isinstance(vis_image, dict):
            # 如果返回的是字典，创建并列显示的图像
            images = list(vis_image.values())
            height = images[0].shape[0]
            width = images[0].shape[1]
            canvas = np.zeros((height, width * len(images), 3), dtype=np.uint8)

            for i, img in enumerate(images):
                if "eyeinhand" in list(vis_image.keys())[i]:
                    img = cv2.flip(cv2.flip(img, 0), 1)
                canvas[:, i * width:(i + 1) * width] = img

            # 在每个图像上方添加标签
            for i, key in enumerate(vis_image.keys()):
                cv2.putText(canvas, key, (i * width + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            cv2.imshow("Simulation", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        else:
            # 如果返回的是单个图像
            cv2.imshow("Simulation", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        cv2.waitKey(2)
    
    def close(self):
        self.sim_env.close()
        self.teleop.disconnect()

    def _key_val_mapping(self, action):
        if isinstance(action, dict):
            action_list = [
                action.get('delta_y', 0.0),
                -action.get('delta_x', 0.0),
                action.get('delta_z', 0.0)
            ]
            
            if self.use_gripper:
                gripper_value = action.get('gripper', 1.0)-1.0
                if gripper_value == 1 or gripper_value==-1:
                    
                    self.last_gripper_value=gripper_value
                action_list.append(self.last_gripper_value)
                
            action_numpy = np.array(action_list, dtype=np.float32)
        else:
            action_numpy = np.array(action, dtype=np.float32)

        return action_numpy

def main():
    key_sim_env = KeySimEnv(use_gripper=True, 
                            display_cameras=True,
                            moving_mode="linear")

    observation, info = key_sim_env.reset()
    key_sim_env.render()

    # record_dataset(policy=None, env=key_sim_env, cfg=cfg)

    while True:
        # 必须获得非0的输入
        action_dict = key_sim_env.teleop.get_action()
        while all(action_dict[key] == default for key, default in 
               [("delta_x", 0), ("delta_y", 0), ("delta_z", 0), ("gripper", 1)]):
            action_dict = key_sim_env.teleop.get_action()

        observation, reward, terminated, truncated, info = key_sim_env.step(action_dict)
        # print(f"reward = {reward:.3f}")
        key_sim_env.render()
        time.sleep(key_sim_env.sim_env.unwrapped.dt)
        # time.sleep(0.2)
        if terminated or truncated:
            observation, info = key_sim_env.reset()
            print("环境重置")

    key_sim_env.close()

if __name__ == "__main__":
    main()