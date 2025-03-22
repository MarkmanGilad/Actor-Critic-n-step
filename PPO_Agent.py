import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import statistics as stat

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.log_probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return batches
    
    def get_arrays(self):
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones)

    def store_memory(self, state, action, log_prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, lr, fc1_dims=256, fc2_dims=512, chkpt=1, optim_step = 100, optim_gamma = 0.9, logger = None):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc1_dims)
        self.fc4 = nn.Linear(fc1_dims, n_actions)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        
        self.checkpoint_file = f'Data/Actor{chkpt}.pth'
        self.optimizer = optim.Adam(self.parameters(), lr=lr)       # without weight_decay
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=optim_step, gamma=optim_gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.logger = logger

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
                
        dist = Categorical(logits=x)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=True))

    def get_all_params_as_list(self):
        params = [p.data.cpu().numpy().flatten() for p in self.parameters()]
        return [param for sublist in params for param in sublist]  # Flatten the nested lists

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=256, fc2_dims=512, chkpt=1, optim_step = 100, optim_gamma = 0.9):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = f'Data/Critic{chkpt}.pth'
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc1_dims)
        self.fc4 = nn.Linear(fc1_dims, 1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()  
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=optim_step, gamma=optim_gamma)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=True))

class PPO_Agent:
    def __init__(self, chkpt, input_dims=88, n_actions=4, logger=None, wandb = None):
        self.gamma = 0.995
        self.n_epochs = 2
        self.batch_size = 16
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.optim_step = 5000
        self.optim_gamma = 0.95
        self.critic_actor_ratio = 0.5
        self.logger = logger
        self.wandb = None   # will be updated by Trainer
        self.learn_step = 0 # counter for number of learning
        self.entropy_coefficient = 0.1
        self.max_entropy_coeff = 0.1
        self.min_entropy_coeff = 0.02
        self.entropy_decay_rate = 0.9996
        self.lmbda = 0.995
        self.clip_epsilon = 0.2

        self.actor = ActorNetwork(input_dims, n_actions, self.lr_actor, chkpt=chkpt, optim_step=self.optim_step, 
                                  optim_gamma=self.optim_gamma, logger=self.logger)
        self.critic = CriticNetwork(input_dims, self.lr_critic, chkpt=chkpt, optim_step=self.optim_step, 
                                    optim_gamma=self.optim_gamma)
        self.memory = Memory(self.batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        state = state.to(self.actor.device)
        with T.no_grad():
            dist = self.actor(state)
            value = self.critic(state)
        action = dist.sample().item()
        log_prob = dist.log_prob(T.tensor(action, device=self.actor.device)).item()
        value = value.item()

        return action, log_prob, value

    def get_value (self, state):
        state = state.to(self.actor.device)
        with T.no_grad():
            value = self.critic(state)
        value = value.item()
        return value

    def calculate_advantage_and_returns(self, reward_arr, val_arr, done_arr, next_val):
        advantage = np.zeros_like(reward_arr, dtype=np.float32)
        running_gae = 0

        for t in reversed(range(len(reward_arr))):
            if t == len(reward_arr) - 1:
                done = done_arr[t]
                next_value = next_val
            else:
                done = done_arr[t + 1]
                next_value = val_arr[t + 1]

            delta = reward_arr[t] + self.gamma * next_value * (1 - done) - val_arr[t]
            running_gae = delta + self.gamma * self.lmbda * (1 - done) * running_gae
            advantage[t] = running_gae

        returns = advantage + val_arr

        # Convert to tensors and move to the appropriate device.
        advantage = T.tensor(advantage).to(self.actor.device)
        returns = T.tensor(returns,dtype=T.float32).to(self.actor.device)

        #region Logging for debugging and monitoring.
        self.advantage_mean = advantage.mean().item()
        self.advantage_std = advantage.std().item()
        self.advantage_norm = advantage.mean()
        self.logger.log('advantage_mean', self.advantage_mean)
        self.logger.log('advantage_std', self.advantage_std)
        self.logger.log('advantage_norm', self.advantage_norm)
        #endregion
        return advantage, returns
        
    def learn(self, next_val):
        #region for login
        actor_losses = []   # for logging
        critic_losses = []  # for logging
        total_losses = []   # for logging
        entropy_values = [] # for logging
        self.learn_step += 1
        #endregion
        
        state_arr, action_arr, log_prob_arr, val_arr, reward_arr, done_arr = self.memory.get_arrays()
        
        # Skip learning if too few samples
        if len(state_arr) < 2:
            print(f"Skipping learning: only {len(state_arr)} samples, need at least 2")
            self.memory.clear_memory()
            return

        # Entropy coefficient decay
        self.entropy_coefficient = max(self.min_entropy_coeff, self.entropy_coefficient * self.entropy_decay_rate)
        
        # Compute advantages and returns using GAE
        advantage, returns = self.calculate_advantage_and_returns(reward_arr, val_arr, done_arr, next_val)

        for i in range(self.n_epochs):
            batches = self.memory.generate_batches()
        
            for batch in batches:
                # Prepare tensors from the batch indices
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                batch_advantage = advantage[batch].to(self.actor.device)
                batch_returns = returns[batch].to(self.critic.device)
                old_log_probs = T.tensor(log_prob_arr[batch]).to(self.actor.device)

                # Compute the current policy distribution and associated log probabilities
                dist = self.actor(states)
                new_log_probs = dist.log_prob(actions)
                critic_value = self.critic(states).squeeze(-1) 
                
                # Compute the probability ratio
                ratio = T.exp(new_log_probs - old_log_probs)
                # Compute the clipped objective
                surr1 = ratio * batch_advantage
                surr2 = T.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)\
                    * batch_advantage
                actor_loss = -T.min(surr1, surr2).mean()

                # Calculate critic loss (value function loss)
                critic_loss = F.mse_loss(critic_value, batch_returns)
                
                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                
                # Combine losses: note that critic_actor_ratio scales the value loss,
                # and the entropy bonus is subtracted (to maximize entropy)
                total_loss = actor_loss + self.critic_actor_ratio * critic_loss \
                    - self.entropy_coefficient * entropy

                #region Logging
                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item()) 
                total_losses.append(total_loss.item())
                entropy_values.append(entropy.item())
                #endregion
                # Perform backward pass and optimization
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        
        self.wandb(values = val_arr.mean(), returns = returns.mean(), advantage=advantage.mean(), 
                   critic_losses= stat.mean(critic_losses), actor_losses=stat.mean(actor_losses), 
                   total_losses= stat.mean(total_losses), entropy= stat.mean(entropy_values))
        self.critic.scheduler.step()
        self.actor.scheduler.step()

        self.memory.clear_memory()
