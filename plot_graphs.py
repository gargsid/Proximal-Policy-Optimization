from matplotlib import pyplot as plt
import gym
import os, sys, cv2 
import numpy as np 
import torch


plt.figure()
x = 1000
# rewards
graphs_root = 'graphs/lunar/'
reinforce = 'graphs/lunar/rewards/reinforce.npy'
reinforce_np = np.load(reinforce)
print('reinforce_np:', reinforce_np.shape)

mean = np.mean(reinforce_np, axis=0)[:x]
std = np.std(reinforce_np, axis=0)[:x]

plt.plot(np.arange(x), mean, label='reinforce')
plt.fill_between(np.arange(x), mean-std, mean+std, alpha=0.2)

# sarsa = 'graphs/lunar/rewards/sarsa.npy'
# sarsa_np = np.load(sarsa)
# print('sarsa_np:', sarsa_np.shape)


# mean = np.mean(sarsa_np, axis=0)[:x]
# std = np.std(sarsa_np, axis=0)[:x]

# plt.plot(np.arange(x), mean, label='sarsa')
# plt.fill_between(np.arange(x), mean-std, mean+std, alpha=0.2)

graphs_root = 'graphs/lunar/'
ppo = 'graphs/lunar/ppo.pth'
ckpt = torch.load(ppo)

ppo_np = np.array(ckpt['graphs']['rewards'])
print('ppo_np:', ppo_np.shape)

mean = np.mean(ppo_np, axis=0)[:x]
std = np.std(ppo_np, axis=0)[:x]

plt.plot(np.arange(x), mean, label='ppo')
plt.fill_between(np.arange(x), mean-std, mean+std, alpha=0.2)

plt.legend(loc='best')

plt.title('mean rewards over 10 runs')
plt.ylabel('rewards')
plt.xlabel('episodes')

path = 'graphs/lunar/rewards.jpg'
plt.savefig(path)
plt.close()

# episode lengths

plt.figure()
x = 999
graphs_root = 'graphs/lunar/'
reinforce = 'graphs/lunar/episode_length/reinforce.npy'
reinforce_np = np.load(reinforce)

reinforce_np = np.diff(reinforce_np, axis=1)

print('reinforce_np:', reinforce_np.shape)

mean = np.mean(reinforce_np, axis=0)[:x]
std = np.std(reinforce_np, axis=0)[:x]

plt.plot(np.arange(x), mean, label='reinforce')
plt.fill_between(np.arange(x), mean-std, mean+std, alpha=0.2)

# sarsa = 'graphs/lunar/episode_length/sarsa.npy'
# sarsa_np = np.load(sarsa)
# sarsa_np = np.diff(sarsa_np, axis=1)

# print('sarsa_np:', sarsa_np.shape)

# mean = np.mean(sarsa_np, axis=0)[:x]
# std = np.std(sarsa_np, axis=0)[:x]

# plt.plot(np.arange(x), mean, label='sarsa')
# plt.fill_between(np.arange(x), mean-std, mean+std, alpha=0.2)

ppo_np = np.array(ckpt['graphs']['episode_length'])
print('ppo_np:', ppo_np.shape)

mean = np.mean(ppo_np, axis=0)[:x]
std = np.std(ppo_np, axis=0)[:x]

plt.plot(np.arange(x), mean, label='ppo')
plt.fill_between(np.arange(x), mean-std, mean+std, alpha=0.2)


plt.legend(loc='best')

plt.title('mean episode lengths over 10 runs')
plt.ylabel('episode length')
plt.xlabel('episodes')

path = 'graphs/lunar/episode_length.jpg'
plt.savefig(path)
plt.close()