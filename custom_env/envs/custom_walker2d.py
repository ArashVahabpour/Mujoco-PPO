from gym.envs.mujoco import Walker2dEnv
import numpy as np
from random import random


class CustomWalker2d(Walker2dEnv):

    def __init__(self):
        self._direction = None
        self._randomize_direction()
        super().__init__()
        self.n_latent = 1  # 1 dimension for sign

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt) * self._direction
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        ob = self._get_obs()
        # if random() < 0.005:
        #     self._randomize_direction()
        return ob, reward, done, {}

    def _randomize_direction(self):
        # +/- 1 indicating going forward or backward and 0 indicating standing still
        self._direction = 2 * np.random.randint(2) - 1  # np.random.randint(3) - 1

    @property
    def latent_code(self):
        return np.array([self._direction])
    
    def reset(self):
        self._randomize_direction()
        return super().reset()
