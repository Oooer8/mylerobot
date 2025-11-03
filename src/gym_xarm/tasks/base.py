import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import _ALL_RENDERERS, MujocoRenderer
from gymnasium_robotics.utils import mujoco_utils

from gym_xarm.tasks import mocap

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") == "glfw":
    RENDER_MODES.append("human")
elif os.environ.get("MUJOCO_GL") not in _ALL_RENDERERS:
    os.environ["MUJOCO_GL"] = "egl"


class Base(gym.Env):
    """
    Superclass for all gym-xarm environments.
    Args:
            xml_name (str): name of the xml environment file
            gripper_rotation (list): initial rotation of the gripper (given as a quaternion)
    """

    metadata = {
        "render_modes": RENDER_MODES,
        "render_fps": 25,
    }
    n_substeps = 20
    initial_qpos = {}
    _mujoco = mujoco
    _utils = mujoco_utils

    def __init__(
        self,
        task,
        obs_type="state",
        render_mode="rgb_array",
        gripper_rotation=None,
        observation_width=84,
        observation_height=84,
        visualization_width=680,
        visualization_height=680,
    ):
        # Coordinates
        if gripper_rotation is None:
            gripper_rotation = [0, 1, 0, 0]
        self.gripper_rotation = np.array(gripper_rotation, dtype=np.float32)

        # Observations
        self.obs_type = obs_type

        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Assets
        self.xml_path = os.path.join(os.path.dirname(__file__), "assets", f"{task}.xml")
        if not os.path.exists(self.xml_path):
            raise OSError(f"File {self.xml_path} does not exist")

        # Initialize sim, spaces & renderers
        self._initialize_simulation()
        self.observation_renderer = self._initialize_renderer(renderer_type="observation")
        self.visualization_renderer = self._initialize_renderer(renderer_type="visualization")
        
        # Get camera names for observation space setup
        self.camera_names = []
        for i in range(self.model.ncam):
            camera_name = self.model.camera(i).name
            if camera_name.startswith("camera"):
                self.camera_names.append(camera_name)
        
        self.observation_space = self._initialize_observation_space()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.metadata["action_space"]),))
        self.action_padding = np.zeros(4 - len(self.metadata["action_space"]), dtype=np.float32)

        assert (
            int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        if "w" not in self.metadata["action_space"]:
            self.action_padding[-1] = 1.0

    def _initialize_simulation(self):
        """Initialize MuJoCo simulation data structures mjModel and mjData."""
        self.model = self._mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.observation_width
        self.model.vis.global_.offheight = self.observation_height

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_time = self.data.time
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()


    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment.

        Can be used to configure initial state and extract information from the simulation.
        """
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        mocap.reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.center_of_table_surf = np.copy(
            self._utils.get_site_xpos(self.model, self.data, "centeroftable"))
        table_half_size = self.model.geom_size[self.model.geom("tablegeom0").id]
        self.center_of_table_surf[2] += table_half_size[2]
        self._sample_goal()
        mujoco.mj_forward(self.model, self.data)

    def _initialize_observation_space(self):
        image_shape = (self.observation_height, self.observation_width, 3)
        
        if self.obs_type == "state":
            # 获取一次观测来确定形状
            obs = self._get_obs()
            observation_space = gym.spaces.Box(-1000.0, 1000.0, shape=obs.shape, dtype=np.float64)
        elif self.obs_type == "pixels":
            # 为每个相机视角创建一个Box空间
            spaces = {}
            for camera_name in self.camera_names:
                spaces[camera_name] = gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
            observation_space = gym.spaces.Dict(spaces)
        elif self.obs_type == "pixels_agent_pos":
            # 为每个相机视角创建一个Box空间
            pixel_spaces = {}
            for camera_name in self.camera_names:
                pixel_spaces[camera_name] = gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
            
            # 获取agent_pos的形状
            agent_pos = self.robot_state
            
            observation_space = gym.spaces.Dict({
                "pixels": gym.spaces.Dict(pixel_spaces),
                "agent_pos": gym.spaces.Box(low=-1000.0, high=1000.0, shape=agent_pos.shape, dtype=np.float64)
            })
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, pixels_agent_pos]"
            )

        return observation_space

    def _initialize_renderer(self, renderer_type: str):
        if renderer_type == "observation":
            model = self.model
        elif renderer_type == "visualization":
            # HACK: gymnasium doesn't allow for custom size rendering on-the-fly, so we
            # initialize another renderer with appropriate size for visualization purposes
            # see https://gymnasium.farama.org/content/migration-guide/#environment-render
            from copy import deepcopy

            model = deepcopy(self.model)
            model.vis.global_.offwidth = self.visualization_width
            model.vis.global_.offheight = self.visualization_height
        else:
            raise ValueError(
                f"Unknown render type {renderer_type}. Must be one of [observation, visualization]"
            )

        return MujocoRenderer(model, self.data)

    @property
    def dt(self):
        """Return the timestep of each Gymanisum step."""
        return self.n_substeps * self.model.opt.timestep

    @property
    def eef(self):
        return self._utils.get_site_xpos(self.model, self.data, "grasp") - self.center_of_table_surf

    @property
    def eef_velp(self):
        return self._utils.get_site_xvelp(self.model, self.data, "grasp") * self.dt

    @property
    def gripper_angle(self):
        return self._utils.get_joint_qpos(self.model, self.data, "right_outer_knuckle_joint")

    @property
    def robot_state(self):
        return np.concatenate([self.eef - self.center_of_table_surf, self.gripper_angle])

    @property
    def obj(self):
        return self._utils.get_site_xpos(self.model, self.data, "object_site") - self.center_of_table_surf

    @property
    def obj_rot(self):
        return self._utils.get_joint_qpos(self.model, self.data, "object_joint0")[-4:]

    @property
    def obj_velp(self):
        return self._utils.get_site_xvelp(self.model, self.data, "object_site") * self.dt

    @property
    def obj_velr(self):
        return self._utils.get_site_xvelr(self.model, self.data, "object_site") * self.dt

    def is_success(self):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        return NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset MuJoCo simulation to initial state.

        Note: Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.

        Args:
            seed (optional integer): The seed that is used to initialize the environment's PRNG (`np_random`). Defaults to None.
            options (optional dictionary): Can be used when `reset` is override for additional information to specify how the environment is reset.

        Returns:
            observation (dictionary) : Observation of the initial state.
            info (dictionary): This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        observation = self.get_obs()
        info = {}
        return observation, info

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.

        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        # max_attempts = 10
        # for attempt in range(max_attempts):
        #     self.data.time = self.initial_time
        #     self.data.qpos[:] = self.initial_qpos.copy()
        #     self.data.qvel[:] = self.initial_qvel.copy()
        #     self._sample_goal()
            
        #     self._mujoco.mj_forward(self.model, self.data)
        #     gripper_pos = self._utils.get_site_xpos(self.model, self.data, "grasp")
        #     print(f"机械臂末端位置: {gripper_pos}")
        #     # 检查是否有严重穿透
        #     has_penetration = False
        #     if self.data.ncon > 0:
        #         for i in range(self.data.ncon):
        #             contact = self.data.contact[i]
        #             # 如果穿透深度 < -0.001（负值表示穿透）
        #             if contact.dist < -0.001:
        #                 has_penetration = True
        #                 break
            
        #     if not has_penetration:
        #         # 没有穿透，可以安全地进行物理模拟
        #         self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        #         return True
        #     else:
        #         print(f"⚠️ 尝试 {attempt + 1}: 检测到穿透，重新采样...")
        #         continue
        
        # # 如果所有尝试都失败，使用方案2（不进行物理模拟）
        # print("⚠️ 无法找到无穿透的初始配置，跳过物理模拟")
        # return True
        self.data.time = self.initial_time
        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()
        self._sample_goal()
        # print("init obj postion", self._utils.get_joint_qpos(self.model, self.data, "object_joint0"))
        self._mujoco.mj_forward(self.model, self.data)
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # print("exec obj postion", self._utils.get_joint_qpos(self.model, self.data, "object_joint0"))
        return True

    def get_obs(self):
        if self.obs_type == "state":
            return self._get_obs()
        
        # 获取所有相机的图像
        pixels = {}
        for camera_name in self.camera_names:
            pixels[camera_name] = self._render(renderer=self.observation_renderer, camera_name=camera_name)
        
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": self.robot_state,
            }
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, pixels_agent_pos]"
            )

    def step(self, action):
        assert action.shape == (4,)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        self._apply_action(action)
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._step_callback()
        observation = self.get_obs()
        reward, reward_components= self.get_reward()
        terminated = is_success = self.is_success()
        truncated = False
        info = {"is_success": is_success,
                "reward_components": reward_components}

        return observation, reward, terminated, truncated, info

    def _step_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _limit_gripper(self, gripper_pos, pos_ctrl):
        rel_pos = gripper_pos - self.center_of_table_surf
        
        x_min, x_max = self.gripper_limit_to_cots['x_range']
        y_min, y_max = self.gripper_limit_to_cots['y_range']
        z_min, z_max = self.gripper_limit_to_cots['z_range']

        if rel_pos[0] > x_max:
            pos_ctrl[0] = min(pos_ctrl[0], 0)
        if rel_pos[0] < x_min:
            pos_ctrl[0] = max(pos_ctrl[0], 0)
        if rel_pos[1] > y_max:
            pos_ctrl[1] = min(pos_ctrl[1], 0)
        if rel_pos[1] < y_min:
            pos_ctrl[1] = max(pos_ctrl[1], 0)
        if rel_pos[2] > z_max:
            pos_ctrl[2] = min(pos_ctrl[2], 0)
        if rel_pos[2] < z_min:
            pos_ctrl[2] = max(pos_ctrl[2], 0)
        return pos_ctrl

    def _apply_action(self, action):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl = self._limit_gripper(
            self._utils.get_site_xpos(self.model, self.data, "grasp"), pos_ctrl
        ) * (1 / self.n_substeps)
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        gripper_ctrl = np.array([gripper_ctrl])
        mocap.apply_action(
            self.model,
            self._model_names,
            self.data,
            np.concatenate([pos_ctrl, self.gripper_rotation, gripper_ctrl]),
        )

    def _set_gripper(self, gripper_pos, gripper_rotation):
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap2", gripper_pos)
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap2", gripper_rotation)
        self._utils.set_joint_qpos(self.model, self.data, "drive_joint", 0)
        self.data.qpos[10] = 0.0
        self.data.qpos[12] = 0.0

    def render(self):
        render_dict = {}
        for camera_name in self.camera_names:
            render_dict[camera_name] = self._render(renderer=self.visualization_renderer, camera_name=camera_name)
        return render_dict

    def _render(self, renderer: MujocoRenderer, camera_name):
        self._render_callback()
        render = renderer.render(self.render_mode, camera_name=camera_name)
        return render.copy() if render is not None else None

    def _render_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def close(self):
        """Close contains the code necessary to "clean up" the environment.

        Terminates any existing WindowViewer instances in the Gymnasium MujocoRenderer.
        """
        if self.observation_renderer is not None:
            self.observation_renderer.close()
        if self.visualization_renderer is not None:
            self.visualization_renderer.close()