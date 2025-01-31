import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

# Device configuration
device = 'cpu'

# Define the Q-Network for Double DQN
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state):
        conv_out = self.conv(state)
        q_values = self.fc(conv_out)
        return q_values

# Define the Double DQN Agent
class DoubleDQN_Agent(object):
    def __init__(self,
                 state_dim=37,
                 num_actions=25,
                 device='cpu',
                 gamma=0.99,
                 tau=0.1,
                 lr=0.0001):
        self.device = device
        self.Q = DQN(state_dim, num_actions).to(device)
        self.Q_target = copy.deepcopy(self.Q)
        self.tau = tau
        self.gamma = gamma
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

    def train(self, batches, epoch):
        (state, next_state, action, next_action, reward, done, bloc_num, SOFAS) = batches
        batch_size = 128
        uids = np.unique(bloc_num)
        num_batches = len(uids) // batch_size

        record_loss = []
        sum_q_loss = 0
        batch_count = 0

        for batch_idx in range(num_batches + 1):
            batch_uids = uids[batch_idx * batch_size: min((batch_idx + 1) * batch_size, len(uids))] 
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

            if batch_count % 25 == 0:
                print('Epoch :', epoch, 'Batch :', batch_count, 'Average Loss :', sum_q_loss / (batch_count + 1))
                record_loss.append(sum_q_loss / (batch_count + 1))
            if batch_count % 100 == 0:
                self.polyak_target_update()

            batch_count += 1

        return record_loss

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(param.data)

    # def compute_loss(self, batch):
    #     state, next_state, action, next_action, reward, done, SOFA = batch
    #     gamma = 0.99
    #     end_multiplier = 1 - done
    #     batch_size = state.shape[0]
    #     range_batch = torch.arange(batch_size).long().to(self.device)

    #     q_values = self.Q(state)
    #     q_values_next = self.Q(next_state)

    #     # Double DQN: select action using the current Q-values, but calculate the next Q-value using the target Q-network
    #     with torch.no_grad():
    #         target_q_values_next = self.Q_target(next_state)

    #     next_action_selected = torch.argmax(q_values_next, dim=1)
    #     target_q = target_q_values_next[range_batch, next_action_selected]

    #     target = reward + gamma * target_q * end_multiplier

    #     q_value_selected = q_values[range_batch, action]
        
    #     loss = F.mse_loss(q_value_selected, target)

    #     return loss

    def compute_loss(self, batch, use_sofa=True):
        state, next_state, action, next_action, reward, done, SOFA = batch
        gamma = 0.99
        end_multiplier = 1 - done
        batch_size = state.shape[0]
        range_batch = torch.arange(batch_size).long().to(self.device)

        # Get the Q-values for the current state-action pair
        log_Q_dist_prediction = self.Q(state.clone().detach().float().to(self.device))
        log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action]

        # Get the Q-values for the next state and the best action using the target network (Double DQN)
        q_eval4nex = self.Q(next_state.clone().detach().float().to(self.device))
        max_eval_next = torch.argmax(q_eval4nex, dim=1)

        with torch.no_grad():
            Q_dist_target = self.Q_target(next_state.clone().detach().float().to(self.device))

        # Compute Q-values based on the target network (using Double DQN's target update mechanism)
        Q_dist_eval = Q_dist_target[range_batch, max_eval_next]
        max_target_next = torch.argmax(Q_dist_target, dim=1)
        Q_dist_tar = Q_dist_target[range_batch, max_target_next]

        # Compute softmax probabilities for the next state-action pair
        Q_target_pro = F.softmax(Q_dist_target, dim=1)
        pro1 = Q_target_pro[range_batch, max_eval_next]
        pro2 = Q_target_pro[range_batch, max_target_next]

        # Linear combination of the two target Q-values based on the probabilities
        Q_dist_star = (pro1 / (pro1 + pro2)) * Q_dist_eval + (pro2 / (pro1 + pro2)) * Q_dist_tar
        log_Q_experience = Q_dist_target[range_batch, next_action.squeeze(1)]

        # If use_sofa flag is True, adjust based on SOFA score
        if use_sofa:
            # With SOFA: Use a conditional check based on SOFA score
            Q_experi = torch.where(SOFA < 4, log_Q_experience, Q_dist_star)
        else:
            # Without SOFA: Use the log_Q_experience directly for all cases
            Q_experi = log_Q_experience

        # Calculate the target Q-value with reward and next state Q-values
        targetQ1 = reward + (gamma * Q_experi * end_multiplier)

        # Compute and return the loss (using Smooth L1 loss as in DDQN)
        return nn.SmoothL1Loss()(targetQ1, log_Q_dist_prediction1)

    def get_action(self, state):
        with torch.no_grad():
            q_values = self.Q(state)
            action = torch.argmax(q_values, dim=1)
            return action
