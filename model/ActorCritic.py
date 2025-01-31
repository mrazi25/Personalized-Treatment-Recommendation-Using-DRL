import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

# Define the Actor-Critic Networks
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(ActorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.network(state)


# Define the Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_dim=37, num_actions=25, device='cpu', gamma=0.99, tau=0.1):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Initialize actor and critic networks
        self.actor = ActorNetwork(state_dim, num_actions).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=0.0001)

    def train(self, batches, epoch):
        (state, next_state, action, next_action, reward, done, bloc_num, SOFAS) = batches
        batch_s = 128
        uids = np.unique(state[:, 0].cpu().numpy())  # Assuming the first column identifies the batch
        num_batch = uids.shape[0] // batch_s

        record_loss = []
        Batch = 0

        for batch_idx in range(num_batch + 1):
            batch_uids = uids[batch_idx * batch_s: min((batch_idx + 1) * batch_s, len(uids))] 
            batch_user = np.isin(state[:, 0].cpu().numpy(), batch_uids)
            state_user = state[batch_user, :]
            next_state_user = next_state[batch_user, :]
            action_user = action[batch_user]
            reward_user = reward[batch_user]
            done_user = done[batch_user]

            batch = (state_user, next_state_user, action_user, reward_user, done_user)
            actor_loss, critic_loss = self.compute_loss(batch)

            # Update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if Batch % 25 == 0:
                print(f"Epoch: {epoch}, Batch: {Batch}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")
                record_loss.append((actor_loss.item(), critic_loss.item()))

            if Batch % 100 == 0:
                self.polyak_target_update()

            Batch += 1

        return record_loss

    def polyak_target_update(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def compute_loss(self, batch):
        state, next_state, action, reward, done = batch
        end_multiplier = 1 - done

        # Critic loss
        state_value = self.critic(state)
        with torch.no_grad():
            next_state_value = self.critic_target(next_state)
        target_value = reward + self.gamma * next_state_value * end_multiplier
        critic_loss = F.smooth_l1_loss(state_value, target_value)

        # Actor loss
        action_probs = self.actor(state)
        
        # To prevent log(0) or log(negative), clamp action probabilities
        action_probs = torch.clamp(action_probs, min=1e-8)  # Add a small value to avoid log(0)
        
        action_log_probs = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        actor_loss = -torch.mean(action_log_probs * (target_value - state_value.detach()))

        return actor_loss, critic_loss

    def get_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
            action = torch.multinomial(action_probs, num_samples=1)
            return action

    def check_for_nans(self, state, next_state, reward, done):
        """ Check if the data contains NaN or Inf values. """
        if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
            print("Detected NaN or Inf in the state input!")
        if torch.any(torch.isnan(next_state)) or torch.any(torch.isinf(next_state)):
            print("Detected NaN or Inf in the next state input!")
        if torch.any(torch.isnan(reward)) or torch.any(torch.isinf(reward)):
            print("Detected NaN or Inf in the reward input!")
        if torch.any(torch.isnan(done)) or torch.any(torch.isinf(done)):
            print("Detected NaN or Inf in the done input!")
