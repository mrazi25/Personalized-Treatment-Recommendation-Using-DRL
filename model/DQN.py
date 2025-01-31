import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

# Device configuration
device = 'cpu'

# Define the Plain DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DQN, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state):
        return self.fc(state)


# Define the DQN Agent
class DQN_Agent:
    def __init__(self, state_dim=37, num_actions=25, device='cpu', gamma=0.99, tau=0.1):
        self.device = device
        self.Q = DQN(state_dim, num_actions).to(device)
        self.Q_target = copy.deepcopy(self.Q)
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.0001)

    def train(self, batches, epoch):
        (state, next_state, action, next_action, reward, done, bloc_num, SOFAS) = batches
        batch_s = 128
        uids = np.unique(bloc_num)
        num_batch = uids.shape[0] // batch_s

        record_loss = []
        sum_q_loss = 0

        record_loss = []
        sum_q_loss = 0

        for batch_idx in range(num_batch + 1):
            batch_uids = uids[batch_idx * batch_s: min((batch_idx + 1) * batch_s, len(uids))]
            batch_user = np.isin(bloc_num, batch_uids)
            state_user = state[batch_user, :]
            next_state_user = next_state[batch_user, :]
            action_user = action[batch_user]
            next_action_user = next_action[batch_user]
            reward_user = reward[batch_user]
            done_user = done[batch_user]
            SOFAS_user = SOFAS[batch_user]

            batch = (state_user, next_state_user, action_user, next_action_user, reward_user, done_user, SOFAS_user)
            loss = self.compute_loss(batch)

            sum_q_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 25 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Avg Loss: {sum_q_loss / (batch_idx + 1)}')
                record_loss.append(sum_q_loss / (batch_idx + 1))

            if batch_idx % 100 == 0:
                self.polyak_target_update()

        return record_loss

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

    def compute_loss(self, batch):
        state, next_state, action, next_action, reward, done, SOFA = batch
        gamma = 0.99
        end_multiplier = 1 - done
        batch_size = state.shape[0]
        range_batch = torch.arange(batch_size).long().to(self.device)

        #log_Q_dist_prediction = self.Q(torch.tensor(state).float().to(self.device))
        log_Q_dist_prediction = self.Q(state.clone().detach().float().to(self.device))
        log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action]

        #q_eval4nex = self.Q(torch.tensor(next_state).float().to(self.device))
        q_eval4nex = self.Q(next_state.clone().detach().float().to(self.device))
        max_eval_next = torch.argmax(q_eval4nex, dim=1)

        with torch.no_grad():
            Q_dist_target = self.Q_target(next_state.clone().detach().float().to(self.device))
            #Q_dist_target = self.Q_target(torch.tensor(next_state).float().to(self.device))

        Q_dist_eval = Q_dist_target[range_batch, max_eval_next]
        max_target_next = torch.argmax(Q_dist_target, dim=1)
        Q_dist_tar = Q_dist_target[range_batch, max_target_next]
        Q_target_pro = F.softmax(Q_dist_target, dim=1)
        pro1 = Q_target_pro[range_batch, max_eval_next]
        pro2 = Q_target_pro[range_batch, max_target_next]

        Q_dist_star = (pro1 / (pro1 + pro2)) * Q_dist_eval + (pro2 / (pro1 + pro2)) * Q_dist_tar
        log_Q_experience = Q_dist_target[range_batch, next_action.squeeze(1)]
        Q_experi = torch.where(SOFA < 4, log_Q_experience, Q_dist_star)
        targetQ1 = reward + (gamma * Q_experi * end_multiplier)

        return nn.SmoothL1Loss()(targetQ1, log_Q_dist_prediction1)

    def get_action(self, state):
        with torch.no_grad():
            q_values = self.Q(state)
            return torch.argmax(q_values, dim=1)
