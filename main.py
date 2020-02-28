import gym
import pybulletgym
from agent import Agent
from train import Train
from play import Play

ENV_NAME = "HopperMuJoCoEnv-v0"
test_env = gym.make(ENV_NAME)

n_states = test_env.observation_space.shape
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
n_actions = test_env.action_space.shape[0]

# stack_shape = (84, 84, 4)
max_steps_per_episode = 10000
max_iter = 15000
actor_lr = 4e-4
critic_lr = 4e-4
epochs = 10
clip_range = 0.3
mini_batch_size = 32

T = 2048

if __name__ == "__main__":
    print(f"number of states:{n_states[0]}\n"
          f"action bounds:{action_bounds}\n"
          f"number of actions:{n_actions}")
    # exit(0)
    env = gym.make(ENV_NAME)

    agent = Agent(n_states=n_states[0],
                  action_bounds=action_bounds,
                  n_actions=n_actions,
                  actor_lr=actor_lr,
                  critic_lr=critic_lr)

    trainer = Train(env=env,
                    agent=agent,
                    horizon=T,
                    max_steps_per_episode=max_steps_per_episode,
                    max_iter=max_iter,
                    epochs=epochs,
                    mini_batch_size=mini_batch_size,
                    epsilon=clip_range
                    )
    trainer.step()

    player = Play(env, agent)
    player.evaluate()