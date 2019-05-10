
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayMemory(object):
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.is_av = False
        self.batch_size = 32

    def push(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def size(self):
        return len(self.memory)

    def is_available(self):
        if len(self.memory) > self.batch_size:
            self.is_av = True
        return self.is_av

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()

        self.n_action = n_action

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1 = Flatten()
        self.fc1 = nn.Linear(16, self.n_action)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.flat1(h)
        h = self.fc1(h)
        return h

class Agent(object):
    def __init__(self, n_action):
        self.n_action = n_action
        self.dqn = DQN(self.n_action)
        self.replay = ReplayMemory(50000)
        # self.dqn.cuda()
        self.gamma = 0.95

    def get_action(self, obs, epsilon):
        if random.random() > epsilon:
            q_value = self.dqn.forward(self.img_to_tensor(obs).unsqueeze(0))
            # print('q_value',q_value)
            action = q_value.max(1)[1].data[0].item()
            # print('action', action)
        else:
            action = random.randint(0, self.n_action-1)
        return action

    def remember(self, state, action, reward, done, next_state):
        self.replay.push(state, action, reward, done, next_state)

    def img_to_tensor(self, img):
        img_tensor = torch.FloatTensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def list_to_batch(self, x):
        # transform a list of image to a batch of tensor [batch size, input channel, width, height]

        temp_batch = self.img_to_tensor(x[0])
        temp_batch = temp_batch.unsqueeze(0)
        for i in range(1, self.replay.batch_size):
            img = self.img_to_tensor(x[i])
            img = img.unsqueeze(0)
            temp_batch = torch.cat([temp_batch, img], dim=0)
        return temp_batch

    def train(self):
        if self.replay.is_available():
            memo = self.replay.sample()

            obs_list = []
            action_list = []
            reward_list = []
            done_list = []
            next_obs_list = []
            for i in range(self.replay.batch_size):
                obs_list.append(memo[i][0])
                action_list.append(memo[i][1])
                reward_list.append(memo[i][2])
                done_list.append(memo[i][3])
                next_obs_list.append(memo[i][4])

            obs_list = self.list_to_batch(obs_list)
            next_obs_list = self.list_to_batch(next_obs_list)

            q_list = self.dqn.forward(obs_list)
            next_q_list = self.dqn.forward(next_obs_list)

            next_q_list_max_v,  next_q_list_max_i = next_q_list.max(1)
            expected_q_value = q_list.clone()
            for i in range(self.replay.batch_size):
                temp_index = next_q_list_max_i[i].item()
                expected_q_value[i][action_list[i]] = reward_list[i] + self.gamma * next_q_list[i][temp_index]

            loss = (q_list - expected_q_value).pow(2).mean()
    
            self.dqn.optimizer.zero_grad()
            loss.backward()
            self.dqn.optimizer.step()




