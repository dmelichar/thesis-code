from gym.envs.registration import register

from safe_agents.envs.lunar import LunarSafe

register(
    id='LunarSafe-v0',
    entry_point='safe_agents.envs:LunarSafe',
    max_episode_steps=1000,
    reward_threshold=200,
)
