from gym import envs
from matplotlib import pyplot as plt
import gym
import os, sys, cv2 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Categorical
from PIL import Image
import torch.nn.functional as F
import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        self.conv_1 = nn.Conv2d(4, 32, 8, 4)
        self.conv_2 = nn.Conv2d(32, 64, 4, 2)
        self.conv_3 = nn.Conv2d(64, 64, 3, 1)
        self.linear = nn.Linear(7*7*64, 512)

        self.pi_theta = nn.Linear(512, 4)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv_1(x))
        x = torch.nn.ReLU()(self.conv_2(x))
        x = torch.nn.ReLU()(self.conv_3(x))
        
        x = x.view((-1, 7*7*64))
        x = torch.nn.ReLU()(self.linear(x))

        x = self.pi_theta(x)
        
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.conv_1 = nn.Conv2d(4, 32, 8, 4)
        self.conv_2 = nn.Conv2d(32, 64, 4, 2)
        self.conv_3 = nn.Conv2d(64, 64, 3, 1)
        self.linear = nn.Linear(7*7*64, 512)

        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv_1(x))
        x = torch.nn.ReLU()(self.conv_2(x))
        x = torch.nn.ReLU()(self.conv_3(x))
        
        x = x.view((-1, 7*7*64))
        x = torch.nn.ReLU()(self.linear(x))

        x = self.value(x).view(-1)
        
        return x

def preprocess_obs(obs):
    obs = obs[20:210, :]
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # print('gray', obs, obs.shape)
    obs = np.squeeze(np.asarray(obs)) / 255.
    # print('gray normalized', obs, obs.shape)
    return obs



env = gym.make('BreakoutNoFrameskip-v4')
print('actions:', env.unwrapped.get_action_meanings())

obs_dim = env.observation_space.shape
print('obs:', obs_dim)

act_dim = env.action_space.n
print('action:', act_dim)

lives = env.unwrapped.ale.lives()
print('lives:', lives)

# obs = env.reset()

# for i in range(10000):
#     # cv2.imwrite('images/render/obs_{}.jpg'.format(i), obs[20:210, :])
#     env.render()
#     action = env.action_space.sample()
#     obs, r, done, _ = env.step(action)
#     # print(r)

#     gray_obs = preprocess_obs(obs)

#     print('lives:', env.unwrapped.ale.lives())
    
#     if done:
#         obs = env.reset()
#         print('done:', done)

# sys.exit()

actor = Actor().to(device)
critic = Critic().to(device) 

if os.path.exists('models/ppo_breakout_actor_critic_v3.pth'):
    checkpoint = torch.load('models/ppo_breakout_actor_critic_v3.pth')
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    print('actor_critic model loaded!')

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 20)
fontScale = 1
color = (255, 255, 255)
thickness = 2

obs_state = np.zeros((1, 4, 84, 84))
size = (160, 210)
fps = 25

# eps_frames = []

frame_idx = 0
path = 'images/breakout_frames/'

total_eps = 1
for i in range(total_eps):
    total_reward = 0

    obs = env.reset()
    remaining_lives = env.unwrapped.ale.lives()

    obs, r, done, _ = env.step(1)
    
    obs_frame = np.copy(obs)

    obs = preprocess_obs(np.copy(obs))
    for j in range(4):
        obs_state[0][j] = np.copy(obs)

    torch_obs = torch.from_numpy(obs_state).float().to(device)
    
    # print('frame:', obs_frame.shape)

    text = 'Rewards:{}'.format(total_reward)
    obs_frame = cv2.putText(obs_frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
    cv2.imwrite(os.path.join(path, '{}.jpg'.format(frame_idx)), cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR))
    frame_idx += 1

    while not done:

        pi_theta = actor(torch_obs).view(-1)
        
        action = torch.argmax(pi_theta).item()
        # print('action:', action.item())
        
        for _ in range(4):
            obs, r, done, info = env.step(action)
            total_reward += r 
            
            obs_frame = np.copy(obs)
            text = 'Rewards:{}'.format(total_reward)
            obs_frame = cv2.putText(obs_frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            cv2.imwrite(os.path.join(path, '{}.jpg'.format(frame_idx)), cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR))
            frame_idx += 1

            if env.unwrapped.ale.lives() < remaining_lives :
                env.step(1)
                remaining_lives = env.unwrapped.ale.lives()

            if done:
                break
        
        if done:
            print('total_reward:', total_reward)
            break
        
        # obs_frame = np.copy(obs)
        obs = preprocess_obs(np.copy(obs))

        obs_state = np.roll(obs_state, shift=-1, axis=1)
        obs_state[0][-1] = np.copy(obs)

        torch_obs = torch.from_numpy(obs_state).to(device).float()
        
        # print('total_reward:', total_reward)
        # total_reward += r

        # text = 'Rewards:{}'.format(total_reward)
        # obs_frame = cv2.putText(obs_frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
        # cv2.imwrite(os.path.join(path, '{}.jpg'.format(frame_idx)), cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR))
        # frame_idx += 1
        
        print('lives:', env.unwrapped.ale.lives(), 'rem:', remaining_lives, 'done:', done, 'total_reward:', total_reward)

        # if remaining_lives < env.unwrapped.ale.lives():
        #     env.step(1)
        #     remaining_lives = env.unwrapped.ale.lives()

        # if done:
            # print('total_reward:', total_reward)
            # break

    # video_name = os.path.join('images', 'breakout_{}.avi'.format(i+1))

    # imageio.mimsave(video_name, eps_frames, fps=25)

    sys.exit()

# out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# for j in range(len(eps_frames)):
#     out.write(eps_frames[j])
# out.release()