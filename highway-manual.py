import gym
import highway_env

env = gym.make('highway-v0')
env.configure({
    "manual_control": True,
#    "action": {
#        "type": "ContinuousAction"
#    }
})
env.reset()

while True:
    env.step(env.action_space.sample())
    env.render()

