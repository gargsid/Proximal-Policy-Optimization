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
    def __init__(self, in_features, n_actions, hid_dim=64):
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
    def __init__(self, in_features, hid_dim=64):
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

env = gym.make('CartPole-v0')

obs = env.reset()

obs_dim = obs.size
print('obs:{}'.format(obs_dim))

act_dim = env.action_space.n
print('action:{}'.format(act_dim))

sys.exit()

actor = Actor(in_features=obs_dim, n_actions=act_dim).to(device)

if os.path.exists('models/ppo_cartpole.pth'):
    checkpoint = torch.load('models/ppo_cartpole.pth', map_location=torch.device('cpu'))
    actor.load_state_dict(checkpoint['actor'])
    print('actor model loaded!')

font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 30)
fontScale = 1
color = (0, 0, 0)
thickness = 2

state = np.zeros((1, 4, 84, 84))
total_eps = 5
size = (160, 210)
fps = 25

eps_frames = []

for i in range(total_eps):
    total_reward = 0

    obs = env.reset()
    obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device).float()
    
    obs_frame = np.ascontiguousarray(env.render(mode="rgb_array"))
    # print('frame:', obs_frame.shape)

    text = 'Rewards:{}'.format(total_reward)
    obs_frame = cv2.putText(obs_frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
    # cv2.imwrite('images/lunar_frame.jpg', cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR))

    eps_frames.append(obs_frame)

    done = False

    while not done:

        pi_theta = actor(obs).view(-1)
        
        action = torch.argmax(pi_theta).item()
        # print('action:', action.item())

        obs, r, done, info = env.step(action)
        obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device).float()
        total_reward += r

        obs_frame = np.ascontiguousarray(env.render(mode="rgb_array"))
        text = 'Rewards:{}'.format(total_reward)
        obs_frame = cv2.putText(obs_frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
        eps_frames.append(obs_frame)

        if done:
            print('total_reward:', total_reward)
            break

    video_name = os.path.join('images', 'cartpole_{}.avi'.format(i+1))

    imageio.mimsave(video_name, eps_frames, fps=25)

    if i == 0:
        for j, img in enumerate(eps_frames):
            cv2.imwrite('images/cartpole/frames/{}.jpg'.format(j+1), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    sys.exit()

# out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# for j in range(len(eps_frames)):
#     out.write(eps_frames[j])
# out.release()