#!/usr/bin/env python

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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.image_writer import safe_stop_image_writer

logging.basicConfig(level=logging.INFO)

from scripts_xarm.env import KeySimEnv

class Config:
    def __init__(self):
        self.name = "xarm_lift_moving_keyboard"
    
        self.repo_id = f"oooer/{self.name}"
        self.dataset_root = f"./outputs/datasets/{self.name}"
        self.num_episodes = 500
        self.fps = 25
        self.push_to_hub = False
        self.task = "xarm_lift"
        self.number_of_steps_after_success = 10
        self.control_step = 300
        self.pretrained_policy_name_or_path = None
        self.use_gripper = True
        self.resume = True  # 添加resume参数，用于继续录制现有数据集


def get_episode_confirmation():
    """在回合结束时提供确认选项"""
    print("\n==== 回合结束 ====")
    print("请选择操作:")
    print("1. 保存此回合")
    print("2. 重新录制此回合")
    print("3. 结束录制")
    
    while True:
        choice = input("请输入选项 (1/2/3): ")
        if choice == '1':
            return "save"
        elif choice == '2':
            return "rerecord"
        elif choice == '3':
            return "stop"
        else:
            print("无效选项，请重新输入")


@safe_stop_image_writer
def record_dataset(policy, env:KeySimEnv, cfg:Config):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    This function runs episodes in the environment and records the observations,
    actions, and results for dataset creation.

    Args:
        env: The environment to record from.
        policy: Optional policy to generate actions (if None, uses teleop).
        cfg: Configuration object containing recording parameters.
    """
    # 设置初始动作
    action = env.action_space.sample() * 0.0

    # 配置动作名称
    action_names = ["delta_x_ee", "delta_y_ee", "delta_z_ee"]
    if cfg.use_gripper:
        action_names.append("gripper_delta")

    # 配置数据集特征
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # 添加图像特征
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "image",
                "shape": env.observation_space[key].shape,
                "names": ["height", "width", "channels"],
            }

    # 创建或加载数据集
    if cfg.resume:
        try:
            dataset = LeRobotDataset(
                cfg.repo_id,
                root=cfg.dataset_root,
            )
            # 启动图像写入器
            if any("image" in key for key in features):
                dataset.start_image_writer(
                    num_processes=1,
                    num_threads=4,
                )
            logging.info(f"Resuming dataset with {dataset.num_episodes} episodes")
        except Exception as e:
            logging.warning(f"Failed to load existing dataset: {e}")
            logging.info("Creating new dataset instead")
            dataset = LeRobotDataset.create(
                cfg.repo_id,
                cfg.fps,
                root=cfg.dataset_root,
                use_videos=False,
                features=features,
                image_writer_processes=1,
                image_writer_threads=4,
            )
    else:
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.dataset_root,
            use_videos=False,
            features=features,
            image_writer_processes=1,
            image_writer_threads=4,
        )

    # 记录回合
    episode_index = dataset.num_episodes  # 从当前数据集的回合数开始
    stop_recording = False
    
    while episode_index < cfg.num_episodes and not stop_recording:
        obs, _ = env.reset()
        step = 0
        log_say(f"Recording episode {episode_index}", play_sounds=False)
        logging.info(f"Recording episode {episode_index}")

        # 跟踪成功状态收集
        success_detected = False
        success_steps_collected = 0
        early_exit = False

        # 运行回合步骤
        while step < cfg.control_step and not early_exit:
            # 获取动作
            action_dict = env.teleop.get_action()
            while all(action_dict[key] == default for key, default in 
               [("delta_x", 0), ("delta_y", 0), ("delta_z", 0), ("gripper", 1)]):
                action_dict = env.teleop.get_action()
            
            # 执行环境步骤
            obs, reward, terminated, truncated, info = env.step(action_dict)
            step += 1

            # 与env中设置相同
            action_numpy = env._key_val_mapping(action_dict)/5

            # 处理观察结果
            recorded_action = {
                "action": action_numpy
            }

            # 处理观察数据
            obs_dict = {
                "observation.state": obs["agent_pos"].astype(np.float32),
            }
            # 添加图像观察
            if "pixels" in obs:
                for key in env.observation_space:
                    if "images" in key:
                        camera_name = key.split(".")[-1]
                        if camera_name in obs["pixels"]:
                            obs_dict[key] = obs["pixels"][camera_name]
            obs_processed = {k: v for k, v in obs_dict.items()}

            # 检查是否检测到成功
            if env.sim_env.is_success() and not success_detected:
                success_detected = True
                logging.info("Success detected! Collecting additional success states.")

            # 添加帧到数据集
            frame = {**obs_processed, **recorded_action}

            # 如果在成功收集阶段，继续将奖励标记为1.0
            if success_detected:
                frame["next.reward"] = np.array([1.0], dtype=np.float32)
            else:
                frame["next.reward"] = np.array([reward], dtype=np.float32)

            # 只有在真正完成时才标记为完成
            frame["next.done"] = np.array([success_detected], dtype=bool)
            
            dataset.add_frame(frame, task=cfg.task)
            env.render()
            time.sleep(env.sim_env.unwrapped.dt)

            # 检查是否应该结束回合
            if success_detected:
                success_steps_collected += 1
                print(f"Success steps collected: {success_steps_collected}")
                
            if success_detected and success_steps_collected >= cfg.number_of_steps_after_success:
                logging.info(f"Collected {success_steps_collected} additional success states")
                early_exit = True
                
            # 检查是否达到了终止条件
            if terminated or truncated:
                early_exit = True

        # 回合结束，提供交互式确认
        confirmation = get_episode_confirmation()
        
        if confirmation == "save":
            logging.info(f"Saving episode {episode_index}")
            dataset.save_episode()
            episode_index += 1
        elif confirmation == "rerecord":
            logging.info(f"Re-recording episode {episode_index}")
            dataset.clear_episode_buffer()
        elif confirmation == "stop":
            logging.info("Stopping recording")
            dataset.clear_episode_buffer()
            stop_recording = True

    # 推送到Hub（如果需要）
    if cfg.push_to_hub:
        dataset.push_to_hub()
    
    return dataset


def main():
    key_sim_env = KeySimEnv(use_gripper=True, 
                            display_cameras=False)
    observation, info = key_sim_env.reset()
    key_sim_env.render()

    cfg = Config()
    
    if cfg.resume:
        print("\n==== 继续录制现有数据集 ====")
    else:
        print("\n==== 创建新数据集 ====")
    
    print("使用键盘控制机械臂进行任务")
    print("每个回合结束后，您将看到确认选项:")
    print("1. 保存此回合")
    print("2. 重新录制此回合")
    print("3. 结束录制")
    print("====================\n")

    record_dataset(policy=None, env=key_sim_env, cfg=cfg)


if __name__ == "__main__":
    main()