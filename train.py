from dataset_config import train_set
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.patches import Rectangle
import numpy as np
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

class HeadMotionEnvironment:
    def __init__(self, data_files):
        self.data_files = data_files
        self.data = []
        self.load_data()
        self.reset()
        
    def load_data(self):
        for data_file in self.data_files:
            with open(data_file, 'r') as file:
                head_motion_data = []
                count = 0
                for line in file:
                    values = line.split()
                    if count % 2 == 0:
                        if len(values) >= 3:  # 적어도 3개 이상의 열이 있는지 확인
                            column_2 = float(values[1])
                            column_3 = float(values[2])
                            column_5 = float(values[4])
                            column_6 = float(values[5])
                            
                            adjusted_longitude = column_3+180
                            adjusted_latitude = column_2 + 90
                            x_coordinate = adjusted_longitude
                            y_coordinate = adjusted_latitude
                            x_pixel = int((x_coordinate / 360) * 3840)
                            y_pixel = int((y_coordinate / 180) * 1920)
                            
                            # 가져온 데이터를 리스트에 추가
                            head_motion_data.append((x_pixel, y_pixel))
                    count += 1
                self.data.append(head_motion_data)

    def reset(self):
        self.t = 0
        self.done = False
        return self._get_observation()

    def step(self, action):
        self.t += 1
        if self.t == len(self.data):
            self.done = True

        user_fov = self._get_observation()
        predicted_fov = action

        reward = self.calculate_overlap(user_fov, predicted_fov)

        return self._get_observation(), reward, self.done

    def _get_observation(self):
        return torch.FloatTensor(self.data[self.t]).unsqueeze(0)

    def calculate_overlap(self, user_fov, predicted_fov):
        user_fov = np.array([user_fov[0] - 540, user_fov[1] - 600,
                             user_fov[0] + 540, user_fov[1] + 600])
        predicted_fov = np.array([predicted_fov[0] - 540, predicted_fov[1] - 600,
                                  predicted_fov[0] + 540, predicted_fov[1] + 600])

        dx = min(user_fov[2], predicted_fov[2]) - max(user_fov[0], predicted_fov[0])
        dy = min(user_fov[3], predicted_fov[3]) - max(user_fov[1], predicted_fov[1])
        overlap_area = dx * dy if (dx >= 0 and dy >= 0) else 0

        total_area = 1080 * 1200
        return overlap_area / total_area

class A3CAgent:
    def __init__(self, model, envs, optimizer, device):
        self.model = model
        self.envs = envs
        self.optimizer = optimizer
        self.device = device

    def train(self, global_model, max_episodes):
        episode_idx = 0
        while episode_idx < max_episodes:
            values, log_probs, rewards, masks = [], [], [], []
            entropy = 0

            for _ in range(20):
                state = self.envs.reset()
                state = torch.FloatTensor(state).to(self.device)
                pi, value, _ = self.model(state)
                dist = Categorical(pi)
                action = dist.sample()

                next_state, reward, done = self.envs.step(action.item())
                next_state = torch.FloatTensor(next_state).to(self.device)
                reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
                done = torch.FloatTensor([done]).unsqueeze(0).to(self.device)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                masks.append(1 - done)

                state = next_state

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            values.append(next_value)

            self.optimizer.zero_grad()
            returns = compute_returns(next_value, rewards, masks)
            loss = compute_loss(returns, log_probs, values, entropy)
            loss.backward()
            self.optimizer.step()

            self.model.load_state_dict(global_model.state_dict())
            episode_idx += 1

        return self.model

class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size):
        super(ActorCritic, self).__init__()
        self.num_actions = num_actions
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, num_actions)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x.view(-1, 1, self.input_size), hidden)
        x = F.relu(x)
        pi = F.softmax(self.fc_pi(x), dim=2)
        v = self.fc_v(x)
        return pi, v, hidden

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def compute_loss(returns, log_probs, values, entropy, value_loss_coef=0.5, entropy_coef=0.01):
    returns = torch.cat(returns)
    log_probs = torch.cat(log_probs)
    values = torch.cat(values)
    advantage = returns.detach() - values
    value_loss = advantage.pow(2).mean()
    policy_loss = -(log_probs * advantage.detach()).mean()
    return value_loss * value_loss_coef + policy_loss - entropy * entropy_coef

# 모델 생성
input_size = 2
num_actions = 2
hidden_size = 128
model = ActorCritic(input_size, num_actions, hidden_size)

# 환경 생성
data_files = ["./HMEM_Data/T00/A380.txt", "./HMEM_Data/T01/A380.txt"]
envs = HeadMotionEnvironment(data_files)

# 옵티마이저 생성 및 학습 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 글로벌 모델 생성
global_model = ActorCritic(input_size, num_actions, hidden_size).to(device)

# 에이전트 생성 및 학습
agent = A3CAgent(model, envs, optimizer, device)
max_episodes = 1000
trained_model = agent.train(global_model, max_episodes)

# 학습된 모델 사용 예시
state = envs.reset()
probs, _ = trained_model(torch.FloatTensor(state).to(device))
dist = Categorical(probs)
action = dist.sample()
next_state, reward, done = envs.step(action.item())

print("Action:", action.item())
print("Next State:", next_state)
print("Reward:", reward)
print("Done:", done)