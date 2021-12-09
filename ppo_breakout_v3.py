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

logs_file = os.path.join('models', 'ppo_lunar_v2_version.txt')
only_print = False 

def logprint(log, logs_file=logs_file):
    print(log, end='')
    if only_print==False:
        with open(logs_file, 'a') as f:
            f.write(log)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logprint('device:{}\n'.format(device))

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
# for i in range(100):
#     # cv2.imwrite('images/render/obs_{}.jpg'.format(i), obs[20:210, :])
#     env.render()
#     action = env.action_space.sample()
#     obs, r, done, _ = env.step(action)
#     print(r)

#     if done:
#         obs = env.reset()

actor = Actor().to(device)
critic = Critic().to(device) 

model_path = 'models/ppo_breakout_actor_critic_v3.pth'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    print('actor_critic model loaded from {}!'.format(model_path))


max_updates = int(1e4)
nupdates = 0. 
init_lr = 2.5e-4

actor_optim = torch.optim.Adam(actor.parameters(), lr=init_lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=init_lr)

N = 1
T = 1024
batch_size = 1024 
K = 4 
clip = 0.1
gamma = 0.99
lam = 0.95

logprint('N:{} T:{} bs:{} lr:{} clip:{}\n'.format(N, T, batch_size, init_lr, clip))

best_rewards = -1e6
save_model_flag = False

episodic_rewards = np.zeros(50)
episodes_completed = 0

cummulative_rew_graph = []

obs_state = np.zeros((1, 4, 84, 84))

while nupdates < max_updates:

    states = torch.empty(size=(0, 4, 84, 84), device=device).float()
    actions = torch.empty(size=(0,), device=device).int()
    action_log_probs = torch.empty(size=(0,), device=device).float()
    rewards = torch.empty(size=(0,), device=device).float()
    values = torch.empty(size=(0,), device=device).float()
    dones = torch.empty(size=(0,), device=device).float()
    advantages = torch.empty(size=(0,), device=device).float()
    returns = torch.empty(size=(0,), device=device).float()

    # print('states:', states)
    # print('actions:', actions)
    obs = env.reset()
        # frames.append(np.copy(obs))

    obs, r, done, _ = env.step(1)

    # print('start done')

    # frames.append(np.copy(obs))
    
    obs = preprocess_obs(np.copy(obs))
    for j in range(4):
        obs_state[0][j] = np.copy(obs)
    torch_obs = torch.from_numpy(obs_state).float().to(device)
    
    eps_r = 0.
    eps_len = 0

    for n in range(N):

        # frames = []


        for t in range(T):

            pi_theta = actor(torch_obs)
            v = critic(torch_obs)

            pi = Categorical(logits=pi_theta)
            # print(pi)

            action = pi.sample()
            log_pr = pi.log_prob(action)
            # print('action:', action)
            # print('log_prob:', log_pr)

            states = torch.cat([states, torch_obs], dim=0)
            actions = torch.cat([actions, action.detach()], dim=0)
            action_log_probs = torch.cat([action_log_probs, log_pr.detach()], dim=0)
            values = torch.cat([values, v.detach()], dim=0)

            # print('states:', states)
            # print('actions:', actions)
            # print('logs:', action_log_probs)
            # print('values:', values)
            rew = 0
            act = action.detach().cpu().numpy()[0]
            for _ in range(4):
                obs, r, done, info = env.step(act)
                # print('eps done:', done, r)
                # frames.append(np.copy(obs))

                rew += r
                # obs = preprocess_obs(obs)
                # dummy_obs = np.maximum(dummy_obs, obs)
                
                if env.unwrapped.ale.lives() < 5:
                    done = True
                if done: 
                    break
            
            obs_state = np.roll(obs_state, shift=-1, axis=1)
            # obs_state[0][-1] = preprocess_obs(obs)
            obs_state[0][-1] = np.copy(preprocess_obs(np.copy(obs)))
            torch_obs = torch.from_numpy(obs_state).float().to(device)

            eps_r += rew
            eps_len += 1
            
            torch_r = torch.from_numpy(np.array([np.sign(rew)])).to(device)
            rewards = torch.cat([rewards, torch_r], dim=0)

            torch_done = torch.from_numpy(np.array([float(done)])).to(device)
            dones = torch.cat([dones, torch_done], dim=0)

            if done:
                episodes_completed += 1
                episodic_rewards = np.roll(episodic_rewards, shift=-1, axis=0)
                episodic_rewards[-1] = eps_r
                
                if episodes_completed < 49:
                    current_reward = np.mean(episodic_rewards[-episodes_completed:])
                else:
                    current_reward = np.mean(episodic_rewards)
                
                cummulative_rew_graph.append(current_reward)                

                # print('eps_r:', eps_r)
                # print('eps_len:', eps_len)
                # video_name = os.path.join('images', 'breakout_{}.avi'.format(eps_len))
                # imageio.mimsave(video_name, frames, fps=100)

                # frames = []

                obs = env.reset()
                # frames.append(np.copy(obs))

                obs, _, _, _ = env.step(1)
                # print('if done:', done)
                # frames.append(np.copy(obs))

                for j in range(4):
                    obs_state[0][j] = np.copy(preprocess_obs(np.copy(obs)))
                torch_obs = torch.from_numpy(obs_state).float().to(device)
                
                eps_r = 0.
                eps_len = 0
            
            # break

        # wrong here rewards_np are not of length T but (n+1) * T  

        rewards_np = rewards.detach().cpu().numpy()[-T:]
        values_np = values.detach().cpu().numpy()[-T:]
        dones_np = dones.detach().cpu().numpy()[-T:]

        single_actor_advantages = np.zeros(T)
        returns_np = np.zeros(T)

        v = critic(torch_obs).detach().cpu().numpy()[0]
        ret = 0
        adv = 0
        for t in reversed(range(T)):
            delta = rewards_np[t] + (1 - dones_np[t]) * gamma * v - values_np[t]
            adv = delta + (1 - dones_np[t]) * gamma * lam * adv

            ret = rewards_np[t] + (1 - dones_np[t]) * gamma * ret 

            single_actor_advantages[t] = adv 
            returns_np[t] = ret

            v = values_np[t]

        single_actor_advantages = torch.from_numpy(single_actor_advantages).to(device)
        advantages = torch.cat([advantages, single_actor_advantages], dim=0)

        returns_np = torch.from_numpy(returns_np).to(device)
        returns = torch.cat([returns, returns_np], dim=0)
        # print('advantages:', advantages.shape)
        # print('returns:', returns.shape)
        
        # sys.exit()

        # break
    advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)
    
    if episodes_completed < 49:
        current_reward = np.mean(episodic_rewards[-episodes_completed:])
        logprint('eps: {} mean_rewards: {}\n'.format(episodes_completed, current_reward))
    
    else:
        current_reward = np.mean(episodic_rewards)
        logprint('eps: {} mean_rewards: {}\n'.format(episodes_completed, current_reward))

    plt.figure()
    x = len(cummulative_rew_graph)
    plt.plot(np.arange(x), np.array(cummulative_rew_graph), label='rewards')
    plt.ylabel('rewards')
    plt.xlabel('episodes')
    plt.title('rewards')
    plt.legend(loc='best')
    plt.savefig('images/breakout/rewards.jpg')
    plt.close()


    if best_rewards < current_reward:
        best_rewards = current_reward 

        logprint('model saved with best rewards:{:.4f}!\n'.format(best_rewards))
        torch.save({
            'actor': actor.state_dict(),
            'critic': critic.state_dict(),
            'graph': cummulative_rew_graph,
        }, model_path)
        

    # train_indices = np.random.randint(low=0, high=N * T, size=N * T)
    train_indices = np.arange(N * T)

    for k in range(K):

        for i in range(N * T // batch_size):

            sample_idx = torch.from_numpy(train_indices[i * batch_size : (i+1) * batch_size]).to(device)
            
            
            batch_states = torch.index_select(states, 0, sample_idx)
            batch_actions = torch.index_select(actions, 0, sample_idx)
            batch_log_probs = torch.index_select(action_log_probs, 0, sample_idx)
            batch_returns = torch.index_select(returns, 0, sample_idx)
            # batch_values = torch.index_select(values, 0, sample_idx)

            batch_advantages = torch.index_select(advantages, 0, sample_idx)
            
            with torch.set_grad_enabled(True):
                pi_theta = actor(batch_states)
                v = critic(batch_states)
            
            pi = Categorical(logits=pi_theta)

            new_log_probs = pi.log_prob(batch_actions)
            # print('new_log_probs:', new_log_probs.shape)

            ratio = torch.exp(new_log_probs - batch_log_probs)
            # print('ratio:', ratio.shape)
            clipped_ratio = torch.clamp(ratio, 1-clip, 1+clip)
            # print('clipped_ratio:', clipped_ratio.shape)

            surr1 = ratio * batch_advantages
            # print('surr1:', surr1.shape)
            surr2 = clipped_ratio * batch_advantages
            # print('surr2:', surr2.shape)
            surr_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = pi.entropy().mean()

            policy_loss = surr_loss - 0.01 * entropy_loss
            
            critic_loss = 0.5 * (v - batch_returns) ** 2 
            # print('critic_loss:', critic_loss.shape)
            
            critic_loss = critic_loss.mean()

            actor_optim.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5, norm_type=2)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5, norm_type=2)
            critic_optim.step()

    nupdates += 1

    decay = 1 - (nupdates / max_updates)
    lr = init_lr * decay

    for pg in actor_optim.param_groups:
        pg['lr'] = lr
    
    for pg in critic_optim.param_groups:
        pg['lr'] = lr
    
    clip = clip * decay 

    # if best_rewards < current_reward:
    #     best_rewards = current_reward 

    #     logprint('model saved with best rewards:{:.4f}!\n'.format(best_rewards))
    #     torch.save({
    #         'actor': actor.state_dict(),
    #         'critic': critic.state_dict(),
    #     }, model_path)