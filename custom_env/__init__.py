from gym.envs.registration import register


register(id='CustomAnt-v2',
         entry_point='custom_env.envs:CustomAnt',
         )
register(id='CustomWalker2d-v2',
         entry_point='custom_env.envs:CustomWalker2d',
         )
