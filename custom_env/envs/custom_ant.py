from gym.envs.mujoco import AntEnv
import numpy as np
import traja


class CustomAnt(AntEnv):
    def __init__(self):
        self.traj = None
        self.step_num = 0
        self._reset_traj()
        self.n_latent = 2
        super().__init__()

    def step(self, a):
        xposbefore = self.get_body_com("torso")[:2]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[:2]
        forward_reward = (xposafter - xposbefore).dot(self.latent_code)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        self.step_num += 1
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    @property
    def latent_code(self):
        if self.step_num >= len(self.traj):
            self._reset_traj()
        return self.traj[self.step_num]

    def reset(self):
        return super().reset()

    def _reset_traj(self):
        traj = traja.generate(n=100000, step_length=1, angular_error_sd=0.1, seed=np.random.randint(100000))
        self.traj = np.diff(np.hstack([traj.x[:, None], traj.y[:, None]]), axis=0)
        self.step_num = 0
