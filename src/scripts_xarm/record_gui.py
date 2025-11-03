#!/usr/bin/env python

import os
os.environ["MUJOCO_GL"] = "glfw"
import logging
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
from tkinter import font as tkfont
import signal
import sys

import gymnasium as gym
import gym_xarm
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.utils.utils import log_say

from scripts_xarm.env import KeySimEnv

logging.basicConfig(level=logging.INFO)

class Config:
    def __init__(self):
        self.name = "xarm_lift_moving_keyboard_static_3camera_fixedstart"
        self.repo_id = f"oooer/{self.name}"
        self.dataset_root = f"./outputs/datasets/{self.name}"
        self.moving_mode = "static"
        self.num_episodes = 500
        self.fps = 25
        self.push_to_hub = False
        self.task = "xarm_lift"
        self.number_of_steps_after_success = 10
        self.control_step = 300
        self.use_gripper = True
        self.resume = True  # Add resume parameter to continue recording existing dataset

class RecorderGUI:
    """Tkinter-based recording GUI interface"""
    
    def __init__(self, master, env, dataset=None):
        self.master = master
        self.env = env
        self.dataset = dataset
        self.master.title("Robot Recording Interface")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure window size
        window_width = 1280
        window_height = 550
        self.master.geometry(f"{window_width}x{window_height}")
        
        # Create custom styles
        self.create_styles()
        
        # Create interface elements
        self.create_widgets()
        
        # Status variables
        self.is_recording = True  # Always recording by default
        self.episode_decision = None
        self.episode_number = 0
        self.success_steps = 0
        self.running = True  # Flag to control update loop
        
        # Start updating camera views
        self.update_camera_views()
        
        # Set window focus
        self.master.focus_set()
    
    def create_styles(self):
        """Create custom styles"""
        # Create large fonts
        self.large_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.medium_font = tkfont.Font(family="Helvetica", size=14)
        
        # Create custom styles
        style = ttk.Style()
        
        # Button styles
        style.configure("Large.TButton", font=self.large_font, padding=(15, 10))
        
        # Label styles
        style.configure("Large.TLabel", font=self.large_font)
        style.configure("Medium.TLabel", font=self.medium_font)
        
        # LabelFrame styles
        style.configure("Large.TLabelframe", font=self.large_font)
        style.configure("Large.TLabelframe.Label", font=self.large_font)
    
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Camera view frame
        self.camera_frame = ttk.Frame(main_frame)
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get camera names
        camera_names = []
        for i in range(self.env.sim_env.model.ncam):
            camera_name = self.env.sim_env.model.camera(i).name
            if camera_name.startswith("camera"):
                camera_names.append(camera_name)
        
        # 定义统一的相机显示尺寸
        # 根据图片比例 640:480 = 4:3 计算合适的 Canvas 尺寸
        CANVAS_WIDTH = 400
        CANVAS_HEIGHT = 300  # 400 * (480/640) = 300，保持 4:3 比例
        
        # Create camera canvases
        self.camera_canvases = {}
        self.camera_images = {}
        cols = min(3, len(camera_names))
        
        for i, name in enumerate(camera_names):
            frame = ttk.LabelFrame(self.camera_frame, text=name, style="Large.TLabelframe")
            row, col = i // cols, i % cols
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # 使用 Canvas，设置为图片的精确比例
            canvas = tk.Canvas(
                frame, 
                width=CANVAS_WIDTH, 
                height=CANVAS_HEIGHT,
                bg='gray20',
                highlightthickness=0
            )
            canvas.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)  # expand=False 防止拉伸
            
            self.camera_canvases[name] = canvas
            self.camera_images[name] = None
        
        # Configure grid weights
        for i in range((len(camera_names) + cols - 1) // cols):
            self.camera_frame.grid_rowconfigure(i, weight=0)  # weight=0 防止垂直拉伸
        for i in range(cols):
            self.camera_frame.grid_columnconfigure(i, weight=1, uniform="col")
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=15)
        
        # Episode control buttons
        self.save_button = ttk.Button(control_frame, text="Save Episode", 
                                    command=lambda: self.set_decision("save"),
                                    state=tk.DISABLED, style="Large.TButton")
        self.save_button.pack(side=tk.LEFT, padx=15)
        
        self.rerecord_button = ttk.Button(control_frame, text="Re-record", 
                                        command=lambda: self.set_decision("rerecord"),
                                        state=tk.DISABLED, style="Large.TButton")
        self.rerecord_button.pack(side=tk.LEFT, padx=15)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Recording", 
                                    command=lambda: self.set_decision("stop"),
                                    state=tk.DISABLED, style="Large.TButton")
        self.stop_button.pack(side=tk.LEFT, padx=15)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=25)
        
        # Exit button
        self.exit_button = ttk.Button(control_frame, text="Exit Program", 
                                    command=self.on_closing,
                                    style="Large.TButton")
        self.exit_button.pack(side=tk.RIGHT, padx=15)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Episode counter
        ttk.Label(status_frame, text="Episode:", style="Large.TLabel").pack(side=tk.LEFT, padx=10)
        self.episode_label = ttk.Label(status_frame, text="0", style="Large.TLabel")
        self.episode_label.pack(side=tk.LEFT, padx=10)
        
        # Success steps
        ttk.Label(status_frame, text="Success Steps:", style="Large.TLabel").pack(side=tk.LEFT, padx=25)
        self.success_label = ttk.Label(status_frame, text="0", style="Large.TLabel")
        self.success_label.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = ttk.Label(status_frame, text="Ready", style="Large.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=15)

    def update_camera_views(self):
        """更新相机视图"""
        if not self.running:
            return
            
        try:
            renders = self.env.sim_env.render()
            
            # 统一目标尺寸
            TARGET_WIDTH = 640
            TARGET_HEIGHT = 480

            for camera_name, image in renders.items():
                if camera_name in self.camera_canvases:
                    canvas = self.camera_canvases[camera_name]
                    
                    # 强制缩放到统一尺寸
                    resized_image = cv2.resize(
                        image, 
                        (TARGET_WIDTH, TARGET_HEIGHT), 
                        interpolation=cv2.INTER_AREA
                    )

                    if "eyeinhand" in camera_name:
                        resized_image = cv2.flip(cv2.flip(resized_image, 0), 1)
                    
                    # 获取 Canvas 尺寸
                    canvas_width = canvas.winfo_width()
                    canvas_height = canvas.winfo_height()
                    
                    if canvas_width <= 1:
                        canvas_width = 400
                    if canvas_height <= 1:
                        canvas_height = 300
                    
                    # 直接缩放到 Canvas 尺寸（因为比例已经匹配）
                    display_image = cv2.resize(
                        resized_image, 
                        (canvas_width, canvas_height), 
                        interpolation=cv2.INTER_AREA
                    )
                    
                    pil_image = Image.fromarray(display_image)
                    photo = ImageTk.PhotoImage(image=pil_image)
                    
                    # 清除 Canvas 并显示图像
                    canvas.delete("all")
                    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                    
                    # 保存图像引用
                    self.camera_images[camera_name] = photo
                        
        except Exception as e:
            logging.error(f"Error updating camera views: {e}")
        
        if self.running:
            self.master.after(40, self.update_camera_views)


    def set_decision(self, decision):
        """Set episode decision"""
        self.episode_decision = decision
        self.enable_decision_buttons(False)
        
        if decision == "save":
            self.status_label.configure(text="Episode Saved")
            self.episode_number += 1
            self.episode_label.configure(text=str(self.episode_number))
        elif decision == "rerecord":
            self.status_label.configure(text="Re-recording")
        elif decision == "stop":
            self.status_label.configure(text="Recording Stopped")
            self.is_recording = False
    
    def enable_decision_buttons(self, enable=True):
        """Enable or disable decision buttons"""
        state = tk.NORMAL if enable else tk.DISABLED
        self.save_button.configure(state=state)
        self.rerecord_button.configure(state=state)
        self.stop_button.configure(state=state)
    
    def wait_for_decision(self):
        """Wait for user decision"""
        self.enable_decision_buttons(True)
        self.status_label.configure(text="Waiting for decision...")
        self.episode_decision = None
        
        # Wait for decision
        while self.episode_decision is None and self.running:
            self.master.update()
            time.sleep(0.1)
        
        return self.episode_decision
    
    def update_success_count(self, count):
        """Update success step count"""
        self.success_steps = count
        self.success_label.configure(text=str(count))
    
    def on_closing(self):
        """Handle window closing"""
        logging.info("Closing program...")
        self.running = False
        
        # Clean up resources
        if self.dataset:
            try:
                # Clear buffer if there's an unsaved episode
                self.dataset.clear_episode_buffer()
                # Stop image writer
                if hasattr(self.dataset, 'image_writer') and self.dataset.image_writer:
                    logging.info("Stopping image writer...")
                    self.dataset.stop_image_writer()
            except Exception as e:
                logging.error(f"Error closing dataset: {e}")
        
        # Close environment
        try:
            if self.env:
                logging.info("Closing environment...")
                self.env.close()
        except Exception as e:
            logging.error(f"Error closing environment: {e}")
        
        # Destroy window
        self.master.destroy()
        
        # Ensure program exits
        logging.info("Program exited")
        os._exit(0)  # Force exit all threads


@safe_stop_image_writer
def record_dataset_with_gui(env, cfg):
    """Record dataset using GUI interface"""
    
    # Set up signal handler
    def signal_handler(sig, frame):
        logging.info("Received interrupt signal, closing...")
        if hasattr(signal_handler, 'gui') and signal_handler.gui:
            signal_handler.gui.on_closing()
        else:
            os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create Tkinter root window
    root = tk.Tk()
    
    # Set DPI scaling
    root.tk.call('tk', 'scaling', 1.5)  # Increase overall scaling of interface elements
    
    dataset = None
    
    try:
        # Configure action names
        action_names = ["delta_x_ee", "delta_y_ee", "delta_z_ee"]
        if cfg.use_gripper:
            action_names.append("gripper_delta")
        
        # Configure dataset features
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
        
        # Add image features
        for key in env.observation_space:
            if "image" in key:
                features[key] = {
                    "dtype": "image",
                    "shape": env.observation_space[key].shape,
                    "names": ["height", "width", "channels"],
                }
        
        # Create or load dataset
        if cfg.resume:
            try:
                dataset = LeRobotDataset(
                    cfg.repo_id,
                    root=cfg.dataset_root,
                )
                # Start image writer
                if any("image" in key for key in features):
                    dataset.start_image_writer(
                        num_processes=1,
                        num_threads=4,
                    )
                logging.info(f"Continuing dataset recording, current episodes: {dataset.num_episodes}")
            except Exception as e:
                logging.warning(f"Failed to load existing dataset: {e}")
                logging.info("Creating new dataset")
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
        
        # Create GUI
        gui = RecorderGUI(root, env, dataset)
        signal_handler.gui = gui  # Add GUI instance to signal handler
        
        # Update GUI episode counter
        gui.episode_number = dataset.num_episodes
        gui.episode_label.configure(text=str(dataset.num_episodes))
        
        # Recording loop
        episode_index = dataset.num_episodes
        stop_recording = False
        # env.reset()
        
        def record_episode():
            nonlocal episode_index, stop_recording

            # Check if recording should continue
            if episode_index >= cfg.num_episodes or stop_recording or not gui.is_recording or not gui.running:
                # Schedule next check
                if gui.running:
                    root.after(100, record_episode)
                return
            
            # Reset environment
            obs, _ = env.reset()
            step = 0
            logging.info(f"Recording episode {episode_index}")
            
            # Update status to show current episode recording
            gui.status_label.configure(text=f"Recording Episode {episode_index}")
            
            # Track success status
            success_detected = False
            success_steps_collected = 0
            early_exit = False
            
            # Run episode steps
            while step < cfg.control_step and not early_exit and gui.is_recording and gui.running:
                # Get action
                action_dict = env.teleop.get_action()
                
                while all(action_dict[key] == default for key, default in 
                       [("delta_x", 0), ("delta_y", 0), ("delta_z", 0), ("gripper", 1)]):
                    action_dict = env.teleop.get_action()
                    root.update()
                    if not gui.running:
                        break
                    # time.sleep(0.01)
                    # if time.time() - wait_start > 5:
                    #     break
                
                if not gui.running:
                    break
                
                # Execute environment step
                obs, reward, terminated, truncated, info = env.step(action_dict)
                step += 1
                
                # Process action
                action_numpy = env._key_val_mapping(action_dict)/5
                
                # Process observation
                recorded_action = {
                    "action": action_numpy
                }
                
                # Process observation data
                obs_dict = {
                    "observation.state": obs["agent_pos"].astype(np.float32),
                }
                # Add image observations
                if "pixels" in obs:
                    for key in env.observation_space:
                        if "image" in key:
                            camera_name = key.split(".")[-1]
                            if camera_name in obs["pixels"]:
                                obs_dict[key] = obs["pixels"][camera_name]
                obs_processed = {k: v for k, v in obs_dict.items()}
                
                # Check if success detected
                if env.sim_env.is_success() and not success_detected:
                    success_detected = True
                    logging.info("Success detected! Collecting additional success states.")
                    # Update status to show success
                    gui.status_label.configure(text=f"Success! Collecting additional states")
                
                # Add frame to dataset
                frame = {**obs_processed, **recorded_action}
                
                # Set reward and done status
                if success_detected:
                    frame["next.reward"] = np.array([1.0], dtype=np.float32)
                else:
                    frame["next.reward"] = np.array([reward], dtype=np.float32)
                
                frame["next.done"] = np.array([success_detected], dtype=bool)
                
                dataset.add_frame(frame, task=cfg.task)
                
                # Update success steps
                if success_detected:
                    success_steps_collected += 1
                    gui.update_success_count(success_steps_collected)
                
                # Check if episode should end
                if success_detected and success_steps_collected >= cfg.number_of_steps_after_success:
                    logging.info(f"Collected {success_steps_collected} additional success states")
                    early_exit = True
                
                # Check termination conditions
                if terminated or truncated:
                    early_exit = True
                
                # Keep GUI responsive
                root.update()
                time.sleep(env.sim_env.unwrapped.dt)
            
            # If GUI is closed, return directly
            if not gui.running:
                return
                
            # Episode ended, wait for decision
            gui.status_label.configure(text=f"Episode {episode_index} completed - Choose action")
            decision = gui.wait_for_decision()
            
            # If GUI is closed, return directly
            if not gui.running:
                return
                
            if decision == "save":
                logging.info(f"Saving episode {episode_index}")
                dataset.save_episode()
                episode_index += 1
                # Reset success counter for next episode
                gui.success_steps = 0
                gui.success_label.configure(text="0")
            elif decision == "rerecord":
                logging.info(f"Re-recording episode {episode_index}")
                dataset.clear_episode_buffer()
                # Reset success counter
                gui.success_steps = 0
                gui.success_label.configure(text="0")
            elif decision == "stop":
                logging.info("Stopping recording")
                dataset.clear_episode_buffer()
                stop_recording = True
            
            # Schedule next episode
            if gui.running:
                root.after(100, record_episode)
        
        # Start recording loop
        root.after(100, record_episode)
        
        # Start main loop
        root.mainloop()
        
        # Push to Hub (if needed)
        if cfg.push_to_hub and dataset:
            dataset.push_to_hub()
        
        return dataset
    
    except Exception as e:
        logging.error(f"Error during recording: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up resources
        if dataset:
            try:
                dataset.clear_episode_buffer()
                if hasattr(dataset, 'image_writer') and dataset.image_writer:
                    dataset.stop_image_writer()
            except:
                pass
        
        return None
    finally:
        # Ensure environment is closed
        try:
            env.close()
        except:
            pass


def main():
    # Set up signal handler
    def signal_handler(sig, frame):
        logging.info("Received interrupt signal, closing...")
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    env = None
    
    try:
        # Configuration
        cfg = Config()
        
        # Create environment
        env = KeySimEnv(use_gripper=True, 
                        display_cameras=False,
                        moving_mode=cfg.moving_mode,
                        )
        # env.reset()
        
        # Display welcome message
        print("\n==== Robot Dataset Recording Tool ====")
        if cfg.resume:
            print("Continuing existing dataset")
        else:
            print("Creating new dataset")
        print("Use keyboard to control the robot arm to complete tasks")
        print("⬅️ ⬆️ ⬇️ ➡️ : Horizontal movement | Left/Right shift: Up/Down | Left/Right ctrl: Open/Close gripper")
        print("====================\n")
        
        # Record dataset using GUI
        record_dataset_with_gui(env, cfg)
    except Exception as e:
        logging.error(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure environment is closed
        if env:
            try:
                env.close()
            except:
                pass
        logging.info("Program exited")


if __name__ == "__main__":
    main()