# %% [markdown]
# # 1. Imports & Global Constants

# %%
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# %%
# setting fixed random seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# setting font size for plots
plt.rcParams['font.size'] = 14

AGE_START = 20
AGE_RETIRE = 150
WAGE_MIN = 0
WAGE_MAX = 100


# %% [markdown]
# # 2. Neural Network

# %%
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# %% [markdown]
# # 3. DQN Agent Classes

# %%
class DQNWorkerAgent:
    def __init__(self, state_dim=2, action_dim=2, gamma=0.95, lr=1e-3, batch_size=64,
                 max_memory=10000, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)
        self.q_network = DQNetwork(input_dim=state_dim, output_dim=action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.target_net = copy.deepcopy(self.q_network)
        self.target_update_freq = 500  
        self.learn_steps = 0
        

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
            return torch.argmax(q_values, dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  
        
        # sample a batch of transitions
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = zip(*batch)

        # convert to tensors
        states_t = torch.FloatTensor(states)       # [batch_size, state_dim]
        actions_t = torch.LongTensor(actions)        # [batch_size]
        rewards_t = torch.FloatTensor(rewards)       # [batch_size]
        next_states_t = torch.FloatTensor(next_states)  # [batch_size, state_dim]
        done_t = torch.FloatTensor(done)           # [batch_size]

        # compute current Q-values
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        self.learn_steps += 1

        # compute next Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            if self.learn_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_network.state_dict())

        # target Q-value
        target_q = rewards_t + (1 - done_t) * self.gamma * next_q

        # loss calculation
        loss = self.loss_fn(current_q, target_q)

        # backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # decay epsilon after each update
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# %%
class DQNFirmAgent:
    def __init__(self, state_dim=1, n_wage_levels=10, gamma=0.95, lr=1e-3, batch_size=64,
                 max_memory=10000, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        
        self.state_dim = state_dim
        self.n_wage_levels = n_wage_levels
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)
        
        self.q_network = DQNetwork(input_dim=state_dim, output_dim=n_wage_levels)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # target network for stability
        self.target_net = copy.deepcopy(self.q_network)
        self.target_update_freq = 500  # update target network every 1000 steps
        self.learn_steps = 0

    def store_transition(self, state, action, reward, next_state, done):
        # Each transition is stored as (state, action, reward, next_state, done)
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        # ε-greedy policy: With probability epsilon, choose a random action.
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_wage_levels)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)  # Shape: [1, state_dim]
            q_values = self.q_network(state_t)
            return torch.argmax(q_values, dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = zip(*batch)
        states_t = torch.FloatTensor(states)       # [batch_size, state_dim]
        actions_t = torch.LongTensor(actions)        # [batch_size]
        rewards_t = torch.FloatTensor(rewards)       # [batch_size]
        next_states_t = torch.FloatTensor(next_states)  # [batch_size, state_dim]
        done_t = torch.FloatTensor(done)           # [batch_size]
        
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        self.learn_steps += 1

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            if self.learn_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_network.state_dict())

        target_q = rewards_t + (1 - done_t) * self.gamma * next_q
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# %% [markdown]
# # 4. Q-Learning Agent Classes

# %%
class WorkerAgent:
    def __init__(self, n_states, n_actions, use_age_based=False, alpha=0.1, gamma=0.95, epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.use_age_based = use_age_based
        self.q_table = np.zeros((n_states * (2 if use_age_based else 1), n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_state(self, wage_offer, env, worker_age=None):
        wage_bins = np.linspace(env.wage_min, env.wage_max, self.n_states + 1)
        wage_state = np.clip(np.digitize(wage_offer, wage_bins) - 1, 0, self.n_states - 1)

        if self.use_age_based and worker_age is not None:
            age_threshold = (env.age_retire - env.age_start) / 2
            age_state = 0 if worker_age < (env.age_start + age_threshold) else 1

            combined_state = wage_state * 2 + age_state
            return np.clip(combined_state, 0, len(self.q_table) - 1)
        else:
            return wage_state

    def choose_action(self, state, episode, n_episodes):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)  # explore
        else:
            return np.argmax(self.q_table[state])  # exploit

    def update_q_table(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# %%
class FirmAgent:
    def __init__(self, n_states, n_wage_levels=10, alpha=0.2, gamma=1.0, epsilon=0.7, epsilon_decay=0.995):
        self.q_table = np.zeros((n_states, n_wage_levels))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.n_wage_levels = n_wage_levels
    
    def discretize_action(self, wage, wage_min, wage_max):
        return min(int((wage - wage_min) / (wage_max - wage_min) * self.n_wage_levels), self.n_wage_levels - 1)

    def choose_action(self, state, episode, n_episodes):
        epsilon = max(0.1, self.epsilon * (1 - episode / n_episodes) * 0.5) 
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_wage_levels)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        if done:
            td_target = reward

        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
    
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

# %% [markdown]
# # 5. McCall Model Environment

# %%
# defining the environment
class McCallModelEnv:
    def __init__(self, wage_min=0, wage_max=100, value_match=200, age_start=20, age_retire=150,
                  unemployment_penalty=4.0, gamma=1., p_exit=0.01, age_based_mode=False, 
                  worker_only_mode=False, firm_only_mode=False, random_mode = False):
        
        self.wage_min = wage_min
        self.wage_max = wage_max
        self.value_match = value_match 

        # age based
        self.age_start = age_start
        self.age_retire = age_retire
        self.unemployment_penalty = unemployment_penalty

        # age based mode
        self.age_based_mode = age_based_mode

        # random probability of exit
        self.p_exit = p_exit

        # worker/firm only mode
        self.worker_only_mode = worker_only_mode
        self.firm_only_mode = firm_only_mode
        self.random_mode = random_mode

        self.gamma = gamma
        self.reset()

    def reset(self):
        self.done = False

        if not self.age_based_mode and (self.worker_only_mode or self.firm_only_mode or self.random_mode):
            self.reservation_wage = self.calculate_reservation_wage(
            gamma=self.gamma,
            wage_min=self.wage_min,
            wage_max=self.wage_max,
            unemployment_penalty=self.unemployment_penalty,
            num_wages=100
        )
            return self.reservation_wage
        
        elif self.age_based_mode:
            self.current_age = self.age_start
            if self.worker_only_mode:
                self.res_wage_table = self.age_based_reservation_wage(mode="worker")
            elif self.firm_only_mode:
                self.res_wage_table = self.age_based_reservation_wage(mode="firm")
            else:
                self.res_wage_table = self.age_based_reservation_wage(mode="worker")
                
            self.reservation_wage = self.res_wage_table[self.current_age]

            return self.current_age
    
    def step(self, wage_offer, worker_accept=None):
        # age-based mode
        if self.age_based_mode:
            if self.firm_only_mode:
                return self._step_age_based_firm(wage_offer)
            
            elif self.worker_only_mode:
                return self._step_age_based_worker(wage_offer, worker_accept)
            # firm-worker interaction
            else:
                return self._step_age_based_interaction(wage_offer, worker_accept)

        # non-age based mode
        elif self.worker_only_mode:
            return self._step_worker_only(wage_offer)
        
        elif self.firm_only_mode:
            return self._step_firm_only(wage_offer)
            
        # firm-worker interaction
        elif self.random_mode:
            return self._step_interaction(wage_offer)

        else:
            raise ValueError("Invalid mode")

    # step helper functions
    def _step_worker_only(self, wage_offer):
        if wage_offer >= self.reservation_wage:
            action = 1
            self.done = True
            reward = wage_offer
        else:
            action = 0
            self.done = False
            reward = 0

        return reward, self.done, action
    
    def _step_firm_only(self, wage_offer):
        if wage_offer >= self.reservation_wage:
            action = 1
            reward = self.value_match - wage_offer
            self.done = True
        else:
            action = 0
            reward = -self.unemployment_penalty
            self.done = False
            if np.random.rand() < self.p_exit:
                self.done = True
        
        return reward, self.done, action

    def _step_interaction(self, wage_offer):
        if wage_offer >= self.reservation_wage:
            action = 1
            firm_reward = self.value_match - wage_offer
            worker_reward = wage_offer
            self.done = True
        else:
            action = 0
            firm_reward, worker_reward = -self.unemployment_penalty
            self.done = False
            if np.random.rand() < self.p_exit:
                self.done = True
        
        return firm_reward, worker_reward, self.done, action

    # step helpers for age based modes
    def _step_age_based_worker(self, wage_offer, worker_accept):
        reservation_wage = self.res_wage_table[self.current_age]
        t = self.current_age - self.age_start  # time step when job is accepted
        T = self.age_retire - self.age_start
        if worker_accept == 1:
            action = 1
            # compute total discounted reward 
            t = self.current_age - self.age_start  # time step when job is accepted
            T = self.age_retire - self.age_start  # total time steps until retirement

            if self.gamma != 1:
                reward_raw = wage_offer * (1 - self.gamma**(T - t)) / (1 - self.gamma)
            else:
                reward_raw = wage_offer * (T - t)

            age_scaling = 0.8 + 0.2 * ((self.current_age - self.age_start) / (self.age_retire - self.age_start))
            reward = reward_raw * age_scaling
            self.done = True
        else:
            action = 0
            # reward = 0
            if self.gamma != 1:
                cont_value = self.res_wage_table[self.current_age] * (1 - self.gamma**(T - t)) / (1 - self.gamma)
            else:
                cont_value = self.res_wage_table[self.current_age] * (T - t)
                
            reward = -self.unemployment_penalty + cont_value
            self.done = False
            
        if not self.done:
            self.current_age += 1
            if self.current_age >= self.age_retire:
                self.done = True
        return reward, self.done, action

    def _step_age_based_firm(self, wage_offer):
        # Use the backward induction table for the reservation wage:
        reservation_wage = self.res_wage_table[self.current_age]
        if wage_offer >= reservation_wage:
            action = 1  # accepted
            firm_reward = self.value_match - wage_offer  # firm gets profit
            self.done = True
        else:
            action = 0  # rejected
            firm_reward = -self.unemployment_penalty
            self.done = self.current_age >= self.age_retire

        # Increment age only if the offer is rejected
        if not self.done:
            self.current_age += 1
            if self.current_age >= self.age_retire:
                self.done = True

        return firm_reward, self.done, action

    def _step_age_based_interaction(self, wage_offer, worker_accept):
        t = self.current_age - self.age_start  # time step when job is accepted
        T = self.age_retire - self.age_start
        # reservation_wage = self.res_wage_table[self.current_age]
        if worker_accept == 1:
            if self.gamma != 1:
                worker_reward_raw = wage_offer * (1 - self.gamma**(T - t)) / (1 - self.gamma)
            else:
                worker_reward_raw = wage_offer * (T - t)

            firm_reward = self.value_match - wage_offer

            age_scaling = 0.5 + 0.5 * ((self.current_age - self.age_start) / (self.age_retire - self.age_start))
            worker_reward = worker_reward_raw * age_scaling

            self.done = True
            action = 1

        else:
            firm_reward = -self.unemployment_penalty
            worker_reward = -self.unemployment_penalty
            self.done = False
            action = 0

        if not self.done:
            self.current_age += 1
            if self.current_age >= self.age_retire:
                self.done = True

        return firm_reward, worker_reward, self.done, action

    def calculate_reservation_wage(
        self,
        gamma: float,
        wage_min: float,
        wage_max: float,
        unemployment_penalty: float,
        num_wages: int = 100,
        tol: float = 1e-3,            # convergence tolerance
        max_iters: int = 1000         # maximum iterations
    ) -> float:
        wages = np.linspace(wage_min, wage_max, num_wages)
        V = np.zeros_like(wages)

        for _ in range(max_iters):
            reject_val = -unemployment_penalty + gamma * V.mean()
            newV = np.maximum(wages, reject_val)
            if np.max(np.abs(newV - V)) < tol:
                break
            V = newV

        # smallest wage where V >= reject_val
        idx = np.where(V >= reject_val)[0]
        return wages[idx[0]] if len(idx) > 0 else wages[-1]
    
    def age_based_reservation_wage(self, mode):
        """
        Precompute the reservation wage for each age from age_start to age_retire - 1.
        Returns a dictionary mapping age to reservation wage.
        """
        res_table = {}
        wage_values = np.linspace(self.wage_min, self.wage_max, 101)
        p = np.ones_like(wage_values) / len(wage_values)
        
        for age in range(self.age_start, self.age_retire):
            T = self.age_retire - age
            V = np.zeros(T+1)
            for t in range(T-1, -1, -1):
                wait_value = -self.unemployment_penalty + self.gamma * V[t+1]
                if mode == "worker":
                    if self.gamma != 1:
                        accept_reward = wage_values * (1 - self.gamma**(T - t)) / (1 - self.gamma)
                    else:
                        accept_reward = wage_values * (T - t)

                    accept_values = np.maximum(accept_reward, wait_value)
                else:  # mode == "firm"
                    accept_reward = wage_values  # or some other payoff logic
                    accept_values = np.maximum(accept_reward, wait_value)
                expected_value = np.sum(p * accept_values)
                V[t] = expected_value
            
            valid_indices = np.where(accept_reward >= wait_value)[0]
            if len(valid_indices) > 0:
                res_wage_for_age = wage_values[valid_indices[0]]
            else:
                res_wage_for_age = wage_values[-1]

            res_table[age] = res_wage_for_age
        
        return res_table

# %% [markdown]
# # 6. Simulations

# %%
def simulate(env, n_episodes, agent=None, firm_agent=None, worker_agent=None, evaluate_after=20, timesteps_per_sim=100):
    if env.age_based_mode:
        if env.worker_only_mode:
            return simulate_age_based_worker(env, n_episodes, agent)
        elif env.firm_only_mode:
            return simulate_age_based_firm(env, n_episodes, agent)
        else:
            return simulate_age_based(env, n_episodes, firm_agent, worker_agent)

    elif env.worker_only_mode:
        return simulate_worker_only(env, n_episodes, agent, evaluate_after, timesteps_per_sim)

    elif env.firm_only_mode:
        return simulate_firm_only(env, n_episodes, agent, evaluate_after, timesteps_per_sim)

    # firm and worker interaction
    elif env.random_mode:
        return simulate_interaction(env, n_episodes, firm_agent=firm, worker_agent=worker)

    else:
        raise ValueError("Invalid mode")

# %%
def simulate_age_based_worker(env, n_episodes, agent):
    worker = agent 
    num_age_steps = env.age_retire - env.age_start
    age_offer_data = []
    acceptance_data = []
    recorded_rw = np.zeros((n_episodes, num_age_steps))

    worker_returns = []
    
    for episode in range(n_episodes):
        env.current_age = env.reset()  # env.reset() returns a random starting age in [age_start, age_retire)
        done = False
        age_index = env.current_age - env.age_start
        step_count = 0

        total_reward = 0

        while not done:
            # firm offers a wage from a Gaussian (firms do not learn)
            base = env.res_wage_table[env.current_age]
            wage_offer = np.random.normal(loc=base, scale=(env.wage_max - env.wage_min)*0.1)
            wage_offer = np.clip(wage_offer, env.wage_min, env.wage_max)
            
            remaining = ((env.age_retire - env.current_age) / (env.age_retire - env.age_start))
            state = [wage_offer / env.wage_max,
                    ((env.current_age - env.age_start) / (env.age_retire - env.age_start)),
                    remaining]

            # agent selects action: 0 = reject, 1 = accept
            action = worker.select_action(state)
            
            # step the environment with the current wage offer
            worker_reward, done, env_action = env.step(wage_offer, worker_accept=action)
            age_offer_data.append((env.current_age, wage_offer))
            acceptance_data.append(env_action)

            total_reward += worker_reward

            # build the next state.
            if not done:
                new_age = env.current_age
                new_remaining = ((env.age_retire - new_age) / (env.age_retire - env.age_start))
                next_state = [wage_offer / env.wage_max,
                            ((new_age - env.age_start) / (env.age_retire - env.age_start)),
                            new_remaining]
            else:
                next_state = [0.0, 0.0, 0.0]
            
            # store the transition in the agent’s memory
            worker.store_transition(state, action, worker_reward, next_state, done)
            worker.train_step()
            
            # record the environment’s reservation wage at the current age
            if env.current_age < env.age_retire:
                recorded_rw[episode, age_index] = env.res_wage_table[env.current_age]
            else:
                # if current_age reached or exceeded age_retire, mark as done
                done = True

            if not done:
                env.current_age = new_age
                if env.current_age < env.age_retire:
                    age_index =env.current_age - env.age_start
                else:
                    done = True
        worker_returns.append(total_reward)                
    return recorded_rw, age_offer_data, acceptance_data, worker_returns

# %%
def simulate_age_based_firm(env, n_episodes, agent):
    firm = agent  
    total_offers = []    
    eval_acceptances = []  
    age_offer_data = []  
    timesteps_per_sim = env.age_retire - env.age_start

    firm_returns = []
    
    for sim in range(n_episodes):
        env.reset()  
        done = False
        
        total_reward = 0

        state = [(env.current_age - env.age_start) / (env.age_retire - env.age_start)]
        
        for t in range(timesteps_per_sim):
            if done:
                break
                
            action_index = firm.select_action(state)
            
            # map the action index to a wage offer
            wage_offer = env.wage_min + action_index * (env.wage_max - env.wage_min) / (firm.n_wage_levels - 1)
            total_offers.append(wage_offer)

            age_offer_data.append((env.current_age, wage_offer))
            
            # step the environment in age-based mode
            firm_reward, done, acceptance = env.step(wage_offer)
            total_reward += firm_reward

            # define next state: update the worker's normalized age
            next_state = ([(env.current_age - env.age_start) / (env.age_retire - env.age_start)]
                        if not done else [0.0])
            d = 1.0 if done else 0.0
            
            firm.store_transition(state, action_index, firm_reward, next_state, d)
            firm.train_step()
            
            if acceptance == 1:
                eval_acceptances.append(action_index)
            
            state = next_state
        
        firm_returns.append(total_reward)
        
    return total_offers, eval_acceptances, firm.q_network, age_offer_data, firm_returns

# %%
def simulate_age_based(env, n_episodes, firm_agent, worker_agent):
    age_wage_data = []
    acceptance_data = []
    firm_episode_returns = []
    worker_episode_returns = []

    timesteps = env.age_retire - env.age_start

    for episode in range(n_episodes):
        env.current_age = env.reset()
        done = False
        firm_total_reward = 0
        worker_total_reward = 0

        # build state for firm (normalized age)
        firm_state = [(env.current_age - env.age_start) / (env.age_retire - env.age_start)]

        for t in range(timesteps):
            if done:
                break

            # firm picks wage offer
            action_index = firm_agent.select_action(firm_state)
            wage_offer = env.wage_min + action_index * (env.wage_max - env.wage_min) / (firm_agent.n_wage_levels - 1)

            # worker picks action
            remaining = ((env.age_retire - env.current_age) / (env.age_retire - env.age_start))
            worker_state = [
                wage_offer / env.wage_max,
                ((env.current_age - env.age_start) / (env.age_retire - env.age_start)),
                remaining
            ]

            worker_action = worker_agent.select_action(worker_state)

            # step environment
            firm_reward, worker_reward, done, worker_action = env.step(wage_offer, worker_accept=worker_action)

            # record for plotting
            age_wage_data.append((env.current_age, wage_offer))
            acceptance_data.append(worker_action)

            firm_total_reward += firm_reward
            worker_total_reward += worker_reward

            # build next state
            if not done:
                next_firm_state = [(env.current_age - env.age_start) / (env.age_retire - env.age_start)]

                new_remaining = ((env.age_retire - env.current_age) / (env.age_retire - env.age_start))
                next_worker_state = [
                    wage_offer / env.wage_max,
                    ((env.current_age - env.age_start) / (env.age_retire - env.age_start)),
                    new_remaining
                ]

            else:
                next_firm_state = [0.0]
                next_worker_state = [0.0, 0.0, 0.0]

            # store transition
            d = float(done)
            firm_agent.store_transition(firm_state, action_index, firm_reward, next_firm_state, d)
            worker_agent.store_transition(worker_state, worker_action, worker_reward, next_worker_state, d)

            # train
            firm_agent.train_step()
            worker_agent.train_step()

            # update state
            firm_state = next_firm_state
    
        firm_episode_returns.append(firm_total_reward)
        worker_episode_returns.append(worker_total_reward)

    return age_wage_data, acceptance_data, firm_episode_returns, worker_episode_returns

# %%
def simulate_worker_only(env, n_episodes, agent, evaluate_after=20, timesteps_per_sim=100):
    worker = agent
    episode_returns = []

    eval_rewards = []
    eval_acceptances = []

    all_wage_offers = []
    all_acceptances = []

    for sim in range(n_episodes):
        env.reset()
        cumulative_reward = 0
        acceptances = []

        wage_offers = np.linspace(env.wage_min, env.wage_max, 100)
        np.random.shuffle(wage_offers)

        all_wage_offers.extend(wage_offers)

        for i in range(timesteps_per_sim):
            wage_offer = wage_offers[i]
            state = worker.get_state(wage_offer, env)

            if sim >= evaluate_after and wage_offer >= env.reservation_wage:
                action_taken = 1
            else:
                action_taken = worker.choose_action(state, sim, n_episodes)

            reward, done, _ = env.step(wage_offer)
            cumulative_reward += reward
            acceptances.append(action_taken)
            all_acceptances.append(action_taken)

            if sim < evaluate_after:
                next_state = worker.get_state(wage_offer, env)
                worker.update_q_table(state, action_taken, reward, next_state, done)
                worker.decay_epsilon()

            if done:
                break
            
        episode_returns.append(cumulative_reward)
        
        if sim == evaluate_after:
            eval_rewards.append(cumulative_reward)
            eval_acceptances.extend(acceptances)


    num_bins = 10
    wage_bins = np.linspace(env.wage_min, env.wage_max, num_bins + 1)

    acceptance_counts = np.zeros(num_bins)
    offer_counts = np.zeros_like(acceptance_counts)
    eval_wage_offers = all_wage_offers[-len(eval_acceptances):]

    for offer, accept in zip(eval_wage_offers, eval_acceptances):
        bin_idx = np.digitize(np.round(offer, 2), np.round(wage_bins, 2)) - 1
        if 0 <= bin_idx < num_bins:
            offer_counts[bin_idx] += 1
            acceptance_counts[bin_idx] += accept

    eval_acceptance_rates = np.divide(
        acceptance_counts,
        offer_counts,
        out=np.zeros_like(acceptance_counts, dtype=float),
        where=offer_counts != 0
    )

    return eval_rewards, eval_acceptance_rates, worker.q_table, episode_returns

# %%
def simulate_firm_only(env, n_episodes, agent, evaluate_after=20, timesteps_per_sim=100):
    firm = agent
    eval_acceptances = []  
    total_offers = []     
    episode_returns = []  

    for sim in range(n_episodes): 
        env.reset()
        cumulative_reward = 0
        firm_offers = np.zeros(firm.n_wage_levels)  
        accepted_firm_offers = np.zeros_like(firm_offers)

        for t in range(timesteps_per_sim):
            state = 0  
            action_index = firm.choose_action(state, sim, n_episodes)

            wage_offer = env.wage_min + action_index * (env.wage_max - env.wage_min) / firm.n_wage_levels
            total_offers.append(wage_offer)

            firm_reward, done, action_taken = env.step(wage_offer)
            cumulative_reward += firm_reward

            if sim < evaluate_after:  
                firm.update_q_table(state, action_index, firm_reward, state, done)
                


            if action_taken == 1:  
                accepted_firm_offers[action_index] += 1

            if done:  
                break

        if sim >= evaluate_after:  
            eval_acceptances.extend(accepted_firm_offers)
            
        episode_returns.append(cumulative_reward)

    wage_bins = np.linspace(env.wage_min, env.wage_max, firm.n_wage_levels + 1)
    wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2
    offer_counts = np.histogram(total_offers, bins=wage_bins)[0]
    acceptance_counts = np.histogram(
        [w for w, a in zip(total_offers, eval_acceptances) if a == 1],
        bins=wage_bins
    )[0]

    eval_acceptance_rates = np.divide(
        acceptance_counts,
        offer_counts,
        out=np.zeros_like(acceptance_counts, dtype=float),
        where=offer_counts != 0
    )

    return total_offers, eval_acceptance_rates, firm.q_table, episode_returns

# %%
def simulate_interaction(env, n_episodes, firm_agent, worker_agent):
    firm = firm_agent
    worker = worker_agent

    firm_offers = np.zeros(firm.n_wage_levels)
    worker_accepts = np.zeros_like(firm_offers)
    firm_profits = []
    firm_episode_returns = []
    worker_episode_returns = []

    for episode in range(n_episodes):
        env.reset()
        done = False
        total_profit = 0
        worker_total_reward = 0
        firm_total_reward = 0

        while not done:
            firm_state = 0
            firm_action_index = firm.choose_action(firm_state, episode, n_episodes)
            wage_offer = env.wage_min + firm_action_index * (env.wage_max - env.wage_min) / firm.n_wage_levels

            firm_offers[firm_action_index] += 1

            worker_state = worker.get_state(wage_offer, env)
            worker_action = worker.choose_action(worker_state, episode, n_episodes)

            if worker_action == 1 and wage_offer >= env.reservation_wage:
                worker_reward = wage_offer
                firm_reward = env.value_match - wage_offer
                action_taken = 1
                done = True
            else:
                worker_reward = -1  
                firm_reward = -env.unemployment_penalty  
                action_taken = 0
                if np.random.rand() < env.p_exit:  
                    done = True
            
            firm_reward = env.value_match - wage_offer if action_taken == 1 else -env.unemployment_penalty
            worker_reward = wage_offer if action_taken == 1 else -env.unemployment_penalty

            firm_total_reward += firm_reward
            worker_total_reward += worker_reward

            next_firm_state = firm_state
            next_worker_state = worker_state

            firm.update_q_table(firm_state, firm_action_index, firm_reward, next_firm_state, done)
            worker.update_q_table(worker_state, worker_action, worker_reward, next_worker_state, done)

            if action_taken == 1:
                worker_accepts[firm_action_index] += 1
                total_profit += firm_reward
        
        firm_profits.append(total_profit)
        firm_episode_returns.append(firm_total_reward)
        worker_episode_returns.append(worker_total_reward)
    
    acceptance_rates = np.divide(worker_accepts, firm_offers, out=np.zeros_like(worker_accepts, dtype=float), where=firm_offers != 0)

    return firm_offers, worker_accepts, acceptance_rates, firm_profits, firm_episode_returns, worker_episode_returns

# %% [markdown]
# # 7. Evaluations

# %%
def evaluate(env, agent, mode, timesteps, simulations):
    env.reset()
    
    eval_wage_offers = []
    eval_acceptances = []

    for sim in range(simulations):
        wage_offers = np.linspace(env.wage_min, env.wage_max, timesteps)
        np.random.shuffle(wage_offers)

        for i in range(timesteps):
            wage_offer = wage_offers[i]

            if mode == "worker":
                action = 1 if wage_offer >= env.reservation_wage else 0
            elif mode == "firm":
                state = 0  
                action = agent.choose_action(state, sim, simulations)
                wage_offer = env.wage_min + action * (env.wage_max - env.wage_min) / agent.n_wage_levels

                action = 1 if wage_offer >= env.reservation_wage else 0
            else:
                raise ValueError(f"Invalid mode: {mode}")

            eval_wage_offers.append(wage_offer)
            eval_acceptances.append(action)

            if action == 1:  
                break

    return eval_wage_offers, eval_acceptances


# %%
def evaluate_policy_over_ages(worker_agent, env, wage_points=100):
    """
    Evaluate the worker's trained DQN policy by checking the lowest wage they accept at each age.
    """
    ages = np.arange(env.age_start, env.age_retire)
    reservation_wages = []
    
    wage_grid = np.linspace(1, env.wage_max, wage_points)
    
    for age in ages:
        found_reservation = env.wage_max  # default to max if no acceptance is found
        norm_age = ((age - env.age_start) / (env.age_retire - env.age_start))
        remaining = ((env.age_retire - age) / (env.age_retire - env.age_start))
        
        for wage in wage_grid:
            norm_wage = wage / env.wage_max
            state = [norm_wage, norm_age, remaining]
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = worker_agent.q_network(state_t)
            action = torch.argmax(q_values, dim=1).item()

            if action == 1:  
                found_reservation = wage
                break
        
        reservation_wages.append(found_reservation)
    return ages, reservation_wages


# %%
def evaluate_firm_policy_over_ages(firm_agent, env, n_wage_levels):
    """
    For each age in the age-based env, compute which discrete wage the firm would pick
    by argmaxing its Q-network over the n_wage_levels actions.
    """
    ages = np.arange(env.age_start, env.age_retire)
    firm_offers = []
    for age in ages:
        norm_age = (age - env.age_start) / (env.age_retire - env.age_start)
        # build the single‐feature state tensor
        state = torch.FloatTensor([norm_age]).unsqueeze(0)       # shape [1,1]
        q_vals = firm_agent.q_network(state)                    # shape [1, n_wage_levels]
        a = torch.argmax(q_vals, dim=1).item()                  # best action index
        
        # map action index back to a dollar wage
        wage = env.wage_min + a * (env.wage_max - env.wage_min) / (n_wage_levels - 1)
        firm_offers.append(wage)
    return ages, firm_offers



# %% [markdown]
# # 8. Main Code

# %%
n_episodes = 1000

# %%
# -------------------- AGE-BASED WORKER-ONLY MODE --------------------

# age-based: low discount factor
age_based_env_low = McCallModelEnv(age_based_mode=True, worker_only_mode=True, firm_only_mode=False, gamma=0.01)
worker_low = DQNWorkerAgent(state_dim=3, action_dim=2, gamma=0.01)
results_low, age_offer_data_low, acceptance_data_low, worker_returns_low = simulate(age_based_env_low, n_episodes, agent=worker_low)

# age-based: high discount factor
age_based_env_high = McCallModelEnv(age_based_mode=True, worker_only_mode=True, firm_only_mode=False, gamma=0.9999)
worker_high = DQNWorkerAgent(state_dim=3, action_dim=2, gamma=0.9999)
results_high, age_offer_data_high, acceptance_data_high, worker_returns_high = simulate(age_based_env_high, n_episodes, agent=worker_high)

# %%
ages_low, res_wages_low = evaluate_policy_over_ages(worker_low, age_based_env_low, wage_points=100)
ages_high, res_wages_high = evaluate_policy_over_ages(worker_high, age_based_env_high, wage_points=100)

plt.figure(figsize=(8,5))
plt.plot(ages_low, res_wages_low, marker='o', label='Low Gamma (0.01)')
plt.plot(ages_high, res_wages_high, marker='o', label='High Gamma (0.9999)')
plt.xlabel("Worker Age")
plt.ylabel("Reservation Wage")
plt.title("Worker Reservation Wage vs. Age")
plt.legend()
plt.show()

# learning curve plot

window = 500
# low-γ run
ma_worker_low  = np.convolve(worker_returns_low,  np.ones(window)/window, mode='valid')

# high-γ run
ma_worker_high = np.convolve(worker_returns_high, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,4))
plt.plot(ma_worker_low,  label="Worker Low γ (500-ep MA)")
plt.plot(ma_worker_high, label="Worker High γ (500-ep MA)")
plt.title("Age-Based Worker-Only Returns")
plt.xlabel("Episode")
plt.ylabel("Return (moving avg)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ---- age-based firm only mode ----

env_low = McCallModelEnv(age_based_mode=True, firm_only_mode=True, gamma=0.01)

env_high = McCallModelEnv(age_based_mode=True, firm_only_mode=True, gamma=0.9999)

firm_agent_low = DQNFirmAgent(state_dim=1, n_wage_levels=100, gamma=0.01, lr=1e-3)
firm_agent_high = DQNFirmAgent(state_dim=1, n_wage_levels=100, gamma=0.9999, lr=1e-3)

# simulations
offers_low, acceptances_low, q_net_low, age_offer_data_low, firm_returns_low = simulate(env_low, n_episodes, firm_agent_low)
offers_high, acceptances_high, q_net_high, age_offer_data_high, firm_returns_high = simulate(env_high, n_episodes, firm_agent_high)

# %%
max_age_low  = max(age for age, _ in age_offer_data_low)
max_age_high = max(age for age, _ in age_offer_data_high)

firm_agent_low.epsilon  = 0.0
firm_agent_high.epsilon = 0.0

ages_full_low,  offers_full_low  = evaluate_firm_policy_over_ages(
    firm_agent_low,  env_low,  firm_agent_low.n_wage_levels)
ages_full_high, offers_full_high = evaluate_firm_policy_over_ages(
    firm_agent_high, env_high, firm_agent_high.n_wage_levels)

offers_full_low  = [
    w if age <= max_age_low  else 0
    for age, w in zip(ages_full_low,  offers_full_low)
]
offers_full_high = [
    w if age <= max_age_high else 0
    for age, w in zip(ages_full_high, offers_full_high)
]

plt.figure(figsize=(8,5))
plt.plot(ages_full_low,  offers_full_low,  marker='o',
         label='Low Gamma (0.01)')
plt.plot(ages_full_high, offers_full_high, marker='o',
         label='High Gamma (0.9999)')
plt.xlabel("Worker Age")
plt.ylabel("Firm Wage Offer")
plt.title("Firm Wage Offers vs. Age")
plt.legend()
plt.show()

# learning plot
window = 500

# low-γ run
ma_firm_low  = np.convolve(firm_returns_low,  np.ones(window)/window, mode='valid')

# high-γ run
ma_firm_high = np.convolve(firm_returns_high, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,4))
plt.plot(ma_firm_low,  label="Firm Low γ (500-ep MA)",  color="tab:blue")
plt.plot(ma_firm_high, label="Firm High γ (500-ep MA)", color="tab:orange")
plt.title("Age-Based Firm-Only Returns")
plt.xlabel("Episode")
plt.ylabel("Return (moving avg)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# -------------------- AGE-BASED FIRM-WORKER INTERACTION MODE --------------------
# low gamma
env_age_low = McCallModelEnv(
    age_based_mode=True,
    worker_only_mode=False,
    firm_only_mode=False,
    gamma=0.01,
    age_start=20,
    age_retire=150
)

firm_agent_low = DQNFirmAgent(
    state_dim=1,
    n_wage_levels=100,
    gamma=0.01,
    lr=1e-3
)

worker_agent_low = DQNWorkerAgent(
    state_dim=3,   # [normalized wage, normalized age]
    action_dim=2,  # accept or reject
    gamma=0.01,
    lr=2e-3
)

# high gamma
env_age_high = McCallModelEnv(
    age_based_mode=True,
    worker_only_mode=False,
    firm_only_mode=False,
    gamma=0.9999,
    age_start=20,
    age_retire=150
)

firm_agent_high = DQNFirmAgent(
    state_dim=1,
    n_wage_levels=100,
    gamma=0.9999,
    lr=1e-3
)

worker_agent_high = DQNWorkerAgent(
    state_dim=3,   # [normalized wage, normalized age]
    action_dim=2,  # accept or reject
    gamma=0.9999,
    lr=2e-3
)

# simulation
age_wage_data_low, acceptance_data_low, firm_returns_low, worker_returns_low = simulate_age_based(
    env_age_low, 
    n_episodes, 
    firm_agent_low, 
    worker_agent_low
    )

age_wage_data_high, acceptance_data_high, firm_returns_high, worker_returns_high = simulate_age_based(
    env_age_high, 
    n_episodes, 
    firm_agent_high, 
    worker_agent_high
    )


# %%
# worker interaction mode

ages_low, res_wages_low = evaluate_policy_over_ages(worker_agent_low, env_age_low, wage_points=100)
ages_high, res_wages_high = evaluate_policy_over_ages(worker_agent_high, env_age_high, wage_points=100)

plt.figure(figsize=(8,5))
plt.plot(ages_low, res_wages_low, marker='o', label='Low Gamma (0.01)')
plt.plot(ages_high, res_wages_high, marker='o', label='High Gamma (0.9999)')
plt.xlabel("Worker Age")
plt.ylabel("Reservation Wage")
plt.title("Worker Reservation Wage vs. Age (Interaction)")
plt.legend()
plt.show()

# %%
# firm interaction mode
max_age_low  = max(age for age, _ in age_wage_data_low)
max_age_high = max(age for age, _ in age_wage_data_high)

firm_agent_low.epsilon  = 0.0
firm_agent_high.epsilon = 0.0

ages_full_low,  offers_full_low  = evaluate_firm_policy_over_ages(
    firm_agent_low,  env_age_low,  firm_agent_low.n_wage_levels)
ages_full_high, offers_full_high = evaluate_firm_policy_over_ages(
    firm_agent_high, env_age_high, firm_agent_high.n_wage_levels)

offers_full_low  = [
    w if age <= max_age_low  else 0
    for age, w in zip(ages_full_low,  offers_full_low)
]
offers_full_high = [
    w if age <= max_age_high else 0
    for age, w in zip(ages_full_high, offers_full_high)
]

plt.figure(figsize=(8,5))
plt.plot(ages_full_low,  offers_full_low,  marker='o',
         label='Low Gamma (0.01)')
plt.plot(ages_full_high, offers_full_high, marker='o',
         label='High Gamma (0.9999)')
plt.xlabel("Worker Age")
plt.ylabel("Firm Wage Offer")
plt.title("Firm Wage Offers vs. Age (Interaction)")
plt.legend()
plt.show()

# alternative learning plots

window = 500

# low-γ run
ma_firm_int_low   = np.convolve(firm_returns_low,   np.ones(window)/window, mode='valid')
ma_worker_int_low = np.convolve(worker_returns_low, np.ones(window)/window, mode='valid')

# high-γ run
ma_firm_int_high   = np.convolve(firm_returns_high,   np.ones(window)/window, mode='valid')
ma_worker_int_high = np.convolve(worker_returns_high, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,4))
plt.plot(ma_firm_int_low,   label="Firm Low γ",   color="tab:blue")
plt.plot(ma_worker_int_low, label="Worker Low γ", color="tab:orange")
plt.plot(ma_firm_int_high,   label="Firm High γ",   linestyle='--', color="tab:blue")
plt.plot(ma_worker_int_high, label="Worker High γ", linestyle='--', color="tab:orange")
plt.title("Age-Based Firm-Worker Interaction Returns")
plt.xlabel("Episode")
plt.ylabel("Return (moving avg)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# -------------------- NON-AGE-BASED MODE --------------------
# environment setup
worker_only_env = McCallModelEnv(worker_only_mode=True)
firm_only_env = McCallModelEnv(firm_only_mode=True, p_exit=0.02)
random_env = McCallModelEnv(random_mode=True, p_exit=0.02) # firm + worker but need to fix a bit (add worker)

# agent setup
firm = FirmAgent(n_states=40, n_wage_levels=20)
worker = WorkerAgent(n_states=150, n_actions=2, use_age_based=False)

# -- simulations --

# worker only
wage_bins, eval_acceptance_rates, worker_q_table, worker_episode_returns = simulate(worker_only_env, n_episodes, agent=worker)
worker_eval_offers, worker_eval_acceptances = evaluate(
    worker_only_env, worker, mode="worker", timesteps=100, simulations=50
)

# firm only
total_offers, acceptance_rates, firm_q_table, firm_episode_returns = simulate(firm_only_env, n_episodes, firm)
firm_eval_offers, firm_eval_acceptances = evaluate(
    firm_only_env, firm, mode="firm", timesteps=100, simulations=60
)

# mixed firm + worker interaction
firm_offers, worker_accepts, acceptance_rates, firm_profits, firm_returns, worker_returns, = simulate(random_env, n_episodes, firm, worker)

print("Worker returns:", len(worker_episode_returns))
print("Firm returns:", len(firm_episode_returns))


# %%
# plot for worker-only acceptance rates
num_bins = 50
wage_bins = np.linspace(worker_only_env.wage_min, worker_only_env.wage_max + 1e-5, num_bins + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

acceptance_counts = np.zeros(num_bins)
offer_counts = np.zeros_like(acceptance_counts)

for offer, accept in zip(worker_eval_offers, worker_eval_acceptances):
    bin_idx = np.digitize(offer, wage_bins) - 1
    if 0 <= bin_idx < num_bins:
        offer_counts[bin_idx] += 1
        acceptance_counts[bin_idx] += accept

eval_acceptance_rates = np.divide(
    acceptance_counts,
    offer_counts,
    out=np.zeros_like(acceptance_counts, dtype=float),
    where=offer_counts != 0
)

plt.figure(figsize=(6, 4))
bar_width = (wage_bins[1] - wage_bins[0]) * 0.4  
plt.bar(
    wage_bin_midpoints, 
    eval_acceptance_rates, 
    width=bar_width, 
    color='green', 
    alpha=0.8, 
    label='Worker Acceptance Rates'
)
plt.axvline(worker_only_env.reservation_wage, color='blue', linestyle='--', label='Reservation Wage')
plt.title('Worker Acceptance Rates')
plt.xlabel('Wage Offered')
plt.ylabel('Acceptance Rate')
plt.legend()
plt.tight_layout()
plt.show()

# learning rate plot

window = 500
ma_worker = np.convolve(worker_episode_returns, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,4))
plt.plot(ma_worker, label='Worker (MA)', color='tab:blue')
plt.ylim(55, 95)
plt.title(f'Worker-Only Returns')
plt.xlabel('Episode')
plt.ylabel('Return (moving avg)')
plt.tight_layout()
plt.show()

# %%
# plot for firm-only wage offers
num_bins = 50
wage_bins = np.linspace(firm_only_env.wage_min, firm_only_env.wage_max + 1e-5, num_bins + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

offer_counts = np.zeros(num_bins)

for offer in total_offers:  
    bin_idx = np.digitize(offer, wage_bins) - 1
    if 0 <= bin_idx < num_bins:
        offer_counts[bin_idx] += 1

plt.figure(figsize=(6, 4))
bar_width = (wage_bins[1] - wage_bins[0]) * 0.8  
plt.bar(
    wage_bin_midpoints, 
    offer_counts, 
    width=bar_width, 
    color='blue', 
    alpha=0.7, 
    label='Total Firm Offers'
)
plt.title('Firm Wage Offer Frequency')
plt.xlabel('Wage Offered')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# learning plot

window = 500
ma_fo = np.convolve(firm_episode_returns, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,4))
plt.plot(ma_fo, color='tab:blue')
plt.ylim(90, 140)
plt.title(f'Firm-Only Returns')
plt.xlabel('Episode')
plt.ylabel('Return (moving avg)')
plt.tight_layout()
plt.show()

# %%
# plot for firm-worker interaction
wage_bins = np.linspace(random_env.wage_min, random_env.wage_max, firm.n_wage_levels + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(wage_bin_midpoints, firm_offers, color='blue', alpha=0.7, label='Total Firm Offers')
ax.set_title('Firm Wage Offer Frequency (Interaction)')
ax.set_xlabel('Wage Offered')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(wage_bin_midpoints, acceptance_rates, color='green', alpha=0.7, label='Worker Acceptance Rates')
ax.set_title('Worker Acceptance Rates (Interaction)')
ax.set_xlabel('Wage Offered')
ax.set_ylabel('Acceptance Rate')
ax.legend()
plt.tight_layout()
plt.show()

# learning rate
window = 500
firm_ma = np.convolve(firm_returns, np.ones(window)/window, mode='valid')
worker_ma = np.convolve(worker_returns, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,4))
plt.plot(firm_ma, label='Firm (500-ep MA)')
plt.plot(worker_ma, label='Worker (500-ep MA)')
plt.title(f'Firm and Worker Returns')
plt.xlabel('Episode')
plt.ylabel('Return (moving average)')
plt.legend()
plt.show()


