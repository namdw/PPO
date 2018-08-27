
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym

env = gym.make('CartPole-v0')
s_dim = len(env.observation_space.high)
a_dim = env.action_space.n
h_dim = 128
l_dim = 64

body = nn.Sequential(
			nn.Linear(s_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, l_dim)
			)

policy = nn.Sequential(
			body,
			nn.ReLU(),
			nn.Linear(l_dim, a_dim),
			nn.Softmax(dim=1)
			)

vf = nn.Sequential(
			body,
			nn.ReLU(),
			nn.Linear(l_dim, 1))

optim_p = optim.RMSprop(policy.parameters(), lr=1e-4)
optim_vf = optim.RMSprop(vf.parameters(), lr=1e-4)
criterion = nn.MSELoss()


n_update = int(1e5)
n_mb = 10
n_step = 32
gamma = 0.9
clip = 0.1

# Variables for status checking
avg_score = 0
score_ct = 0
max_score_ct = 20
episodes = 0
steps = 0

state_ = env.reset()
score = 0
for update in range(n_update):
	V = []
	V_ = []
	Ratio = []
	Ent = []
	for mb in range(n_mb):
		traj = []
		for step in range(n_step):
			state = state_
			logit = policy(torch.Tensor(state).unsqueeze(0)).squeeze()
			m = Categorical(logit)
			action = m.sample().numpy()
			state_, reward, done, _ = env.step(action)
			value = vf(torch.Tensor(state).unsqueeze(0)).squeeze().detach().numpy()
			score += reward
			adv = 0
			traj.append((state, action, reward, value))
			if done:
				traj.append((state_, action, 0, 0))
				avg_score += score
				score_ct += 1
				episodes += 1
				steps += step + 1
				if score_ct==max_score_ct:
					print()
					print('-'*20)
					print('Num updates:', update)
					print('Episodes:', episodes)
					print('Steps', steps)
					print('Avg score:', avg_score / max_score_ct)
					print('-'*20)
					avg_score = 0
					score_ct = 0
				# reset the env and other variables and break
				state_ = env.reset()
				score = 0
				break

		batch_size = len(traj)-1
		s_batch = torch.zeros(batch_size, s_dim)
		a_batch = torch.zeros(batch_size, 1)
		v_batch_ = torch.zeros(batch_size, 1)

		v = traj[-1][-1]
		for i, tr in enumerate(reversed(traj[:-1])):
			tr_list = list(tr)
			v = tr[2] + gamma * v
			tr_list.append(v)
			traj[-2-i] = tuple(tr_list)

			s_batch[-1-i] = torch.Tensor(traj[-2-i][0])
			a_batch[-1-i] = torch.Tensor(traj[-2-i][1]).unsqueeze(0)
			v_batch_[-1-i] = torch.Tensor([traj[-2-i][4]]).unsqueeze(0)
		v_batch = vf(s_batch)
		A_batch = v_batch_ - v_batch
		prob_batch = policy(s_batch)
		ent_batch = -(prob_batch * torch.log(prob_batch)).sum(dim=1)
		p_batch = prob_batch.gather(1, a_batch.long())
		ratio1 = p_batch/p_batch.detach()
		ratio2 = ratio1.clamp(1-clip, 1+clip)
		ratio = torch.min(ratio1, ratio2)

		Ratio.append(ratio)
		V.append(v_batch)
		V_.append(v_batch_)
		Ent.append(ent_batch)

		traj = [traj[-1]]

	Ratio = torch.cat(Ratio, dim=0)
	V = torch.cat(V, dim=0)
	V_ = torch.cat(V_, dim=0)
	Ent = torch.cat(Ent, dim=0)

	A = V_ - V
	A = (A - A.mean()) / A.std()
	loss = -A.detach() * Ratio
	loss_v = 0.5*((V_-V).clamp(-clip, clip))**2
	loss = loss.mean() + 0.1 * loss_v.mean() + 0.01 * Ent.mean()
	optim_p.zero_grad()
	optim_vf.zero_grad()
	loss.backward()
	optim_p.step()
	optim_vf.step()





