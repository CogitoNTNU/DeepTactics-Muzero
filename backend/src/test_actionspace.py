
import gymnasium as gym
class Environment(object):
    """The environment MuZero is interacting with."""
    def __init__(self, gamefile: str): #'ALE/Breakout-v5'
        print(gym.envs.registry.keys())
        self.env = gym.make(gamefile)#"""render_mode='human',""" 
        self.obs, self.info = self.env.reset()
        self.episode_over: bool = False
        self.input_size = self.env.action_space
        print(self.input_size)

    def step(self, action):
        self.action = action
        self.obs, self.reward, terminal, truncated, info = self.env.step(action)        
        self.episode_over = terminal or truncated
        return self.reward, self.obs

    def close(self):
        self.env.close()
    
    def action_space(self):
        self.env.action_space

tic = Environment(gamefile='tictactoe-v0')
