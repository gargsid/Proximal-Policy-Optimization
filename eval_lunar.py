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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Actor(nn.Module):
    def __init__(self, in_features, n_actions, hid_dim=128):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(in_features, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, hid_dim)
        self.l4 = nn.Linear(hid_dim, n_actions)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)

        return x 

class Critic(nn.Module):
    def __init__(self, in_features, hid_dim=128):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(in_features, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, hid_dim)
        self.l4 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = x.view(-1)
        return x

env = gym.make('LunarLander-v2')

obs = env.reset()

obs_dim = obs.size
print('obs:{}'.format(obs_dim))

act_dim = env.action_space.n
print('action:{}'.format(act_dim))

actor = Actor(in_features=obs_dim, n_actions=act_dim).to(device)

if os.path.exists('models/ppo_lunar_separate_v2.pth'):
    checkpoint = torch.load('models/ppo_lunar_separate_v2.pth')
    actor.load_state_dict(checkpoint['actor'])
    print('actor model loaded!')

state = np.zeros((1, 4, 84, 84))
total_eps = 5
size = (160, 210)
fps = 25

eps_frames = []

for i in range(total_eps):
    total_reward = 0

    obs = env.reset()
    obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
    
    obs_frame = np.ascontiguousarray(env.render(mode="rgb_array"))
    # print('frame:', obs_frame.shape)

    # sys.exit()

    eps_frames.append(obs_frame)
    print_r_on_frame(eps_frames[-1], total_reward)

    while not done:

        pi_theta = actor(torch.from_numpy(state).float().to(device)).view(-1)
        
        action = torch.argmax(pi_theta)
        action = action.detach().cpu().numpy()[0]

        obs, r, done, info = env.step(action)
        obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
        total_reward += r

        eps_frames.append(np.ascontiguousarray(env.render(mode="rgb_array"))
        print_r_on_frame(eps_frames[-1], total_reward)

        if done:
            print('total_reward:', total_reward)
            break


video_name = os.path.join('images', 'render', 'lunar_lander.avi')

out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for j in range(len(eps_frames)):
    out.write(eps_frames[j])
out.release()