import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import copy

# Device configuration
device = 'cpu'

# Define the Distributional DQN Network
class DistributionalDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc_val = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state):
        conv_out = self.conv(state)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)


# Define the DQN Wrapper
class DDQN_Agent(object):
    def __init__(self,
                 state_dim=37,
                 num_actions=25,
                 device='cpu',
                 gamma=0.999,
                 tau=0.1):
        self.device = device
        self.Q = DistributionalDQN(state_dim, num_actions).to(device)
        self.Q_target = copy.deepcopy(self.Q)
        self.tau = tau
        self.gamma = gamma
        self.num_actions = num_actions
        #self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)

    def train(self, batches, epoch):
        (state, next_state, action, next_action, reward, done, bloc_num, SOFAS) = batches
        batch_s = 128
        uids = np.unique(bloc_num)
        num_batch = uids.shape[0] // batch_s

        record_loss = []
        sum_q_loss = 0
        Batch = 0

        all_train_actions = []  # To store all actions during training

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
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()

            # Save actions for analysis
            current_actions = self.get_action(state_user)
            all_train_actions.append(current_actions.clone().detach().cpu())

            # Save actions for the current batch
            #current_actions = self.get_action(torch.tensor(state_user).float().to(self.device))
            # current_actions = self.get_action(state_user.clone().detach().float().to(self.device))

            # all_train_actions.append(current_actions.clone().detach().cpu())

            if Batch % 25 == 0:
                print('Epoch :', epoch, 'Batch :', Batch, 'Average Loss :', sum_q_loss / (Batch + 1))
                record_loss.append(sum_q_loss / (Batch + 1))
            if Batch % 100 == 0:
                self.polyak_target_update()

            Batch += 1

        # Save all actions at the end of training
        all_train_actions = torch.cat(all_train_actions, dim=0).numpy()
        #np.save(f'train_actions_epoch_{epoch}.npy', all_train_actions)

        return record_loss

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(param.data)

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
            Q_dist = self.Q(state)
            a_star = torch.argmax(Q_dist, dim=1)
            return a_star
