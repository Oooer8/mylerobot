import numpy as np

from gym_xarm.tasks import Base

MOVING_MODE = {
    "static",
    "linear", 
    #"circular", 
    "random"
}

class LiftMoving(Base):
    metadata = {
        **Base.metadata,
        "action_space": "xyzw",
        "episode_length": 50,
        "description": "Lift a moving cube above a height threshold. Moving mode can change as linear, circular and random",
    }

    def __init__(self, **kwargs):
        self._z_threshold = 0.15
        self.moving_mode = kwargs.pop('moving_mode', 'linear')
        self.gripper_limit_to_cots = {
            "x_range": [0.2, 0.6], 
            "y_range": [-0.4, 0.4], 
            "z_range": [0.02, 0.6]
        }

        if self.moving_mode not in MOVING_MODE:
            raise ValueError(f"Moving mode {self.moving_mode} not supported. Choose from {MOVING_MODE}")

        super().__init__("lift_moving", **kwargs)

    @property
    def z_target(self):
        return self._init_z + self._z_threshold - self.center_of_table_surf[2]

    def is_success(self):
        return self.obj[2] >= self.z_target

    def get_reward(self):
        reach_dist = np.linalg.norm(self.obj - self.eef)
        reach_dist_xy = np.linalg.norm(self.obj[:-1] - self.eef[:-1])
        pick_completed = self.obj[2] >= (self.z_target-0.01)
        obj_dropped = (self.obj[2] < (self._init_z + 0.005)) and (reach_dist > 0.02)

        # Reach
        if reach_dist < 0.05:
            reach_reward = -reach_dist + max(self._action[-1], 0) / 50
        elif reach_dist_xy < 0.05:
            reach_reward = -reach_dist
        else:
            z_bonus = np.linalg.norm(np.linalg.norm(self.obj[-1] - self.eef[-1]))
            reach_reward = -reach_dist - 2 * z_bonus

        # Pick
        if pick_completed and not obj_dropped:
            pick_reward = self.z_target
        elif (reach_dist < 0.1) and (self.obj[2] > (self._init_z + 0.005)):
            pick_reward = min(self.z_target, self.obj[2])
        else:
            pick_reward = 0

        reward_components = {
            "reach_reward": reach_reward,
            "pick_reward": pick_reward
        }

        return reach_reward / 100 + pick_reward, reward_components

    def _get_obs(self):
        return np.concatenate(
            [
                self.eef,
                self.eef_velp,
                self.obj,
                self.obj_rot,
                self.obj_velp,
                self.obj_velr,
                self.eef - self.obj,
                np.array(
                    [
                        np.linalg.norm(self.eef - self.obj),
                        np.linalg.norm(self.eef[:-1] - self.obj[:-1]),
                        self.z_target,
                        self.z_target - self.obj[-1],
                        self.z_target - self.eef[-1],
                    ]
                ),
                self.gripper_angle,
            ],
            axis=0,
        )

    def _initialize_object_trajectory_every_episode(self):
        if self.moving_mode == 'static':
            return
        elif self.moving_mode == 'random':
            self.obj_change_steps = np.random.randint(25, 50) 
            self.obj_step_counter = 0

        # initial velocity 
        direction = np.random.uniform(-1, 1, size=2) # X-Y plane
        direction = direction / np.linalg.norm(direction)
        speed = np.random.uniform(0.02, 0.1) # m/s
        self.obj_target_velocity = np.array([direction[0], direction[1], 0.0]) * speed

    def _check_boundary_collision(self):
        """检查物体是否碰到边界并计算反弹方向"""
        x_min, x_max = self.gripper_limit_to_cots['x_range']
        y_min, y_max = self.gripper_limit_to_cots['y_range']
        
        obj_pos_relative = self.obj - self.center_of_table_surf
        
        bounce_x, bounce_y = False, False
        
        if obj_pos_relative[0] <= x_min or obj_pos_relative[0] >= x_max:
            bounce_x = True

        if obj_pos_relative[1] <= y_min or obj_pos_relative[1] >= y_max:
            bounce_y = True
            
        return bounce_x, bounce_y

    def _update_object_velocity_every_step(self):
        # if obj not on the table, stop it
        if self.obj[2] >= self._init_z-self.center_of_table_surf[2]:
            return
        if self.moving_mode == 'static':
            return
        target_vel = np.zeros(6) # obj target vel [vx, vy, vz, wx, wy, wz]

        bounce_x, bounce_y = self._check_boundary_collision()
        
        # 如果碰到边界，反转相应方向的速度
        if bounce_x or bounce_y:
            if bounce_x:
                self.obj_target_velocity[0] = -self.obj_target_velocity[0]
            if bounce_y:
                self.obj_target_velocity[1] = -self.obj_target_velocity[1]
            
            # 可以稍微随机化反弹后的速度方向，使运动更自然
            if self.moving_mode == 'random':
                # 添加一点随机扰动
                perturbation = np.random.uniform(-0.2, 0.2, size=2)
                direction = self.obj_target_velocity[:2] + perturbation
                direction = direction / np.linalg.norm(direction)
                speed = np.linalg.norm(self.obj_target_velocity[:2])
                self.obj_target_velocity[:2] = direction * speed

        if self.moving_mode == 'random':
            self.obj_step_counter += 1
            if self.obj_step_counter >= self.obj_change_steps:
                # create new velocity
                direction = np.random.uniform(-1, 1, size=2)
                direction = direction / np.linalg.norm(direction)
                speed = np.random.uniform(0.02, 0.1)
                self.obj_target_velocity = np.array([direction[0], direction[1], 0.0]) * speed
                self.obj_step_counter = 0
                target_vel[:3] = self._utils.get_joint_qvel(self.model, self.data, "object_joint0")[:3]

        target_vel[:3] += self.obj_target_velocity
        self._utils.set_joint_qvel(self.model, self.data, "object_joint0", target_vel)


    def _sample_goal(self):
        x_min, x_max = self.gripper_limit_to_cots['x_range']
        y_min, y_max = self.gripper_limit_to_cots['y_range']
        z_min, z_max = self.gripper_limit_to_cots['z_range']

        # Gripper
        gripper_pos = self.center_of_table_surf + \
            np.array([
                self.np_random.uniform(x_min, x_max),
                self.np_random.uniform(y_min, y_max),
                self.np_random.uniform(z_min+0.1, z_max)
            ])
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object
        obj_min, obj_max = max(x_min, y_min), min(x_max, y_max)
        object_half_size = self.model.geom_size[self.model.geom("object0").id]
        object_pos = np.copy(self.center_of_table_surf)
        object_pos[0] += self.np_random.uniform(obj_min, obj_max)
        object_pos[1] += self.np_random.uniform(obj_min, obj_max)
        object_pos[2] += object_half_size[2]+5*1e-3

        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)
        self._init_z = object_pos[2]

        # Goal
        return object_pos + np.array([0, 0, self._z_threshold])

    def reset(
        self,
        seed=None,
        options: dict | None = None,
    ):
        self._action = np.zeros(4)
        obs, info = super().reset(seed=seed, options=options)
        
        self._initialize_object_trajectory_every_episode()
        
        return obs, info

    def step(self, action):
        self._update_object_velocity_every_step()
        self._action = action.copy()
        print("obj", self.obj)
        return super().step(action)