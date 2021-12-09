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
logprint('obs:{}\n'.format(obs_dim))

act_dim = env.action_space.n
logprint('action:{}\n'.format(act_dim))

# obs = env.reset()
# for i in range(100):
#     # cv2.imwrite('images/render/obs_{}.jpg'.format(i), obs[20:210, :])
#     env.render()
#     action = env.action_space.sample()
#     obs, r, done, _ = env.step(action)
#     print(r)

#     if done:
#         obs = env.reset()

actor = Actor(in_features=obs_dim, n_actions=act_dim).to(device)
critic = Critic(in_features=obs_dim).to(device)

init_lr = 1e-3

actor_optim = torch.optim.Adam(actor.parameters(), lr=init_lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=init_lr)


N = 1
T = 4096
batch_size = 256 
K = 4 
clip = 0.2
gamma = 0.99
lam = 0.95
reward_scale = 20.

logprint('N:{} T:{} bs:{} lr:{} clip:{}\n'.format(N, T, batch_size, init_lr, clip))

save_model_flag = False

episodic_rewards = np.zeros(50)
episodes_completed = 0

max_episodes = 5000
max_runs = 10

graphs = {
    'rewards': [[] for _ in range(max_runs)], 
    'episode_length': [[] for _ in range(max_runs)], 
    'entropy': [[] for _ in range(max_runs)], 
    'critic_loss': [[] for _ in range(max_runs)],
    'policy_loss': [[] for _ in range(max_runs)],
    'KL_divergence': [[] for _ in range(max_runs)],
    'x_ticks_episodes': [[] for _ in range(max_runs)],
    # 'learning_rate': np.zeros((runs, max_episodes)),
}

for run in range(max_runs):

    logprint('PPO run:{}\n'.format(run))

    target_reward = 300
    current_reward = 0 

    save_model_flag = False

    episodic_rewards = np.zeros(50)
    episodes_completed = 0

    actor = Actor(in_features=obs_dim, n_actions=act_dim).to(device)
    critic = Critic(in_features=obs_dim).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=init_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=init_lr)


    while episodes_completed < max_episodes:

        states = torch.empty(size=(0, obs_dim), device=device).float()
        actions = torch.empty(size=(0,), device=device).int()
        action_log_probs = torch.empty(size=(0,), device=device).float()
        rewards = torch.empty(size=(0,), device=device).float()
        values = torch.empty(size=(0,), device=device).float()
        dones = torch.empty(size=(0,), device=device).float()
        advantages = torch.empty(size=(0,), device=device).float()
        returns = torch.empty(size=(0,), device=device).float()

        # print('states:', states)
        # print('actions:', actions)

        for n in range(N):

            obs = env.reset()
            obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
            eps_r = 0.
            eps_len = 0

            for t in range(T):

                pi_theta = actor(obs)
                v = critic(obs)

                # print(pi_theta)
                # print(v)

                pi = Categorical(logits=pi_theta)
                # print(pi)

                action = pi.sample()
                log_pr = pi.log_prob(action)

                # print('action:', action)
                # print('log_prob:', log_pr)

                states = torch.cat([states, obs], dim=0)
                actions = torch.cat([actions, action.detach()], dim=0)
                action_log_probs = torch.cat([action_log_probs, log_pr.detach()], dim=0)
                values = torch.cat([values, v.detach()], dim=0)

                # print('states:', states)
                # print('actions:', actions)
                # print('logs:', action_log_probs)
                # print('values:', values)

                obs, r, done, info = env.step(action.detach().cpu().numpy()[0])
                obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
                eps_r += r
                eps_len += 1
                
                torch_r = torch.from_numpy(np.array([r / reward_scale])).to(device)
                rewards = torch.cat([rewards, torch_r], dim=0)

                torch_done = torch.from_numpy(np.array([float(done)])).to(device)
                dones = torch.cat([dones, torch_done], dim=0)

                if done:
                    obs = env.reset()
                    obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)

                    episodic_rewards = np.roll(episodic_rewards, shift=-1, axis=0)
                    episodic_rewards[-1] = eps_r

                    graphs['rewards'][run].append(eps_r) 
                    graphs['episode_length'][run].append(eps_len)

                    if eps_r >= target_reward:
                
                        logprint('model saved with best rewards:{:.4f}!\n'.format(eps_r))
                        torch.save({
                            'actor': actor.state_dict(),
                            'critic': critic.state_dict(),
                        }, 'models/ppo_lunar_separate_v2.pth')

                    episodes_completed += 1
                    eps_r = 0.
                    eps_len = 0
                # break

            # wrong here rewards_np are not of length T but (n+1) * T  

            rewards_np = rewards.detach().cpu().numpy()[-T:]
            values_np = values.detach().cpu().numpy()[-T:]
            dones_np = dones.detach().cpu().numpy()[-T:]

            single_actor_advantages = np.zeros(T)
            returns_np = np.zeros(T)

            v = critic(obs).detach().cpu().numpy()[0]
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

        current_reward = np.mean(episodic_rewards)
        logprint('eps: {} mean_rewards: {}\n'.format(episodes_completed, np.mean(episodic_rewards)))
        
        # train_indices = np.random.randint(low=0, high=N * T, size=N * T)
        train_indices = np.arange(N * T)
        
        avg_policy_loss = 0.
        avg_entropy = 0.
        avg_critic_loss = 0.
        avg_kl_div = 0.

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

                kl_div = torch.exp(batch_log_probs) * (batch_log_probs - new_log_probs)
                kl_div = kl_div.mean()

                avg_policy_loss += surr_loss.item() / K
                avg_entropy += entropy_loss.item() / K
                avg_critic_loss += critic_loss.item() / K
                avg_kl_div += kl_div.item() / K

        graphs['policy_loss'][run].append(avg_policy_loss)
        graphs['entropy'][run].append(avg_entropy)
        graphs['critic_loss'][run].append(avg_critic_loss)
        graphs['KL_divergence'][run].append(avg_kl_div)
        graphs['x_ticks_episodes'][run].append(episodes_completed)

    #     break
    # break

rewards = graphs['rewards']
episode_length = graphs['episode_length']

min_x = 1e6
for v in rewards:
    min_x = min(min_x, len(v))
print(min_x)

for i in range(len(rewards)):
    rewards[i] = rewards[i][:min_x]
    episode_length[i] = episode_length[i][:min_x]

entropy = graphs['entropy']
critic_loss = graphs['critic_loss']
policy_loss = graphs['policy_loss']
KL_divergence = graphs['KL_divergence']

min_x = 1e6
for v in entropy:
    min_x = min(min_x, len(v))
print(min_x)

for i in range(len(rewards)):
    entropy[i] = entropy[i][:min_x]
    critic_loss[i] = critic_loss[i][:min_x]
    policy_loss[i] = policy_loss[i][:min_x]
    KL_divergence[i] = KL_divergence[i][:min_x]



for k, v in graphs.items():
    if k == 'x_ticks_episodes':
        continue

    v = np.array(v)
    # print(k)
    # for vv in v:
    #     print(len(vv))

    # sys.exit()

    v_mean = np.mean(v, axis=0)
    v_std = np.std(v, axis=0)
    
    x = len(v_mean)

    plt.figure()
    plt.plot(np.arange(x), v_mean, label=k)
    plt.fill_between(np.arange(x), v_mean-v_std, v_mean+v_std, alpha=0.2)

    plt.legend(loc='best')

    plt.title(k)
    plt.ylabel(k)

    if k=='rewards' or k=='episode_length':
        plt.xlabel('episodes')
    else:
        plt.xlabel('PPO iterations')

    path = os.path.join('images', 'lunar_lander', '{}.jpg'.format(k))
    plt.savefig(path)
    plt.close()

torch.save({
    'graphs': graphs,
}, 'graphs/lunar/ppo.pth')