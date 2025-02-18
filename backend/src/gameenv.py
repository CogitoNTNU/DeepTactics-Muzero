import gymnasium as gym
import ale_py

class game:
    def __init__(self, gamefile: str = 'ALE/Breakout-v5'):
        self.env = gym.make(gamefile, continuous = True)  # remove render_mode in training
        self.obs, self.info = self.env.reset()
        self.episode_over: bool = False

    def register_action(self, move):
        action = self.env.action_space.sample() #move, now its random
        self.obs, self.reward, terminated, truncated, info = self.env.step(action)
        self.episode_over = terminated or truncated
    
    def action_loop(self, moveset):
        while not self.episode_over:
            self.register_action()
        self.env.close()