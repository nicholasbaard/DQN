import gymnasium as gym
import numpy as np

class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward=-1, max_reward=1):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)

def prepare_env(env_name:str="ALE/Breakout-v5", 
                frame_skip:int=4, 
                screen_size:int=84, 
                render_mode:str=None
                ):
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env=env, 
                                          frame_skip=frame_skip, 
                                          screen_size=screen_size, 
                                          scale_obs=True
                                          )
    env = gym.wrappers.FrameStack(env=env, num_stack=4)
    env = ClipReward(env)

    return env

        
if __name__ == "__main__":

    env = prepare_env(env_name="ALE/Breakout-v5", render_mode="human")

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.shape)
    print(env.action_space.n)
    print(env.reward_range)
    # stop
    
    # env.reset()
    # done = False
    # for i in range(2):
    #     done = False
    #     while not done:
    #         env.render()
    #         action = env.action_space.sample()
    #         obs, reward, done, truncated, info = env.step(action)
    #         print(obs.shape)
    #         print(reward)
    #         print(done)
    #         print(info)
    #         if done:
    #             env.reset()