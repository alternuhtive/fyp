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

        # 0416: adding target network for stability
        self.target_net = copy.deepcopy(self.q_network)
        self.target_update_freq = 500  # update target network every 1000 steps
        

    def store_transition(self, state, action, reward, next_state, done):
        # state and next_state are lists or numpy arrays of length state_dim.
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        # Epsilon-greedy: state is a list or numpy array.
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)  # shape [1, state_dim]
            q_values = self.q_network(state_t)
            return torch.argmax(q_values, dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  # not enough samples
        
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

        # compute next Q-values
        with torch.no_grad():
            # next_q = self.q_network(next_states_t).max(dim=1)[0]
            # 0416: using target network for next Q-value
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            # 0416: updating target network
            if self.target_update_freq % 500 == 0:
                self.target_net.load_state_dict(self.q_network.state_dict())

        # target Q-value
        target_q = rewards_t + (1 - done_t) * self.gamma * next_q

        # loss calculation
        loss = self.loss_fn(current_q, target_q)
        print(f"Training loss: {loss.item()}")

        # backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        # 0416: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon after each update
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
            return  # Not enough samples to train.
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.FloatTensor(states)       # [batch_size, state_dim]
        actions_t = torch.LongTensor(actions)        # [batch_size]
        rewards_t = torch.FloatTensor(rewards)       # [batch_size]
        next_states_t = torch.FloatTensor(next_states)  # [batch_size, state_dim]
        dones_t = torch.FloatTensor(dones)           # [batch_size]
        
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_network(next_states_t).max(dim=1)[0]
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
        
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon after each update.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # print(f"Loss: {loss.item():.5f}, Epsilon: {self.epsilon:.5f}")

# %%
# defining the environment
class McCallModelEnv:
    def __init__(self, wage_min=0, wage_max=100, value_match=200, age_start=20, age_retire=150,
                  unemployment_penalty=4.0, gamma=1., p_exit=0.01, age_based_mode=False, 
                  worker_only_mode=False, firm_only_mode=False, random_mode = False):
        
        self.wage_min = wage_min
        self.wage_max = wage_max
        self.value_match = value_match # total val of a match (firm + worker split)

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
            # print(f"Calculated Reservation Wage: {self.reservation_wage}")
            return self.reservation_wage
        
        elif self.age_based_mode:
            # self.current_age = np.random.randint(self.age_start, self.age_retire) # randomize age
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
        # age based mode
        if self.age_based_mode:
            if self.firm_only_mode:
                # Use the backward induction table for the reservation wage:
                reservation_wage = self.res_wage_table[self.current_age]
                if wage_offer >= reservation_wage:
                    action = 1  # accepted
                    firm_reward = self.value_match - wage_offer  # firm gets profit
                    self.done = True
                else:
                    action = 0  # rejected
                    firm_reward = -self.unemployment_penalty
                    self.done = False

                # Increment age only if the offer is rejected
                if not self.done:
                    self.current_age += 1
                    if self.current_age >= self.age_retire:
                        self.done = True

                return firm_reward, self.done, action
            
            elif self.worker_only_mode:
                reservation_wage = self.res_wage_table[self.current_age]
                t = self.current_age - self.age_start  # time step when job is accepted
                T = self.age_retire - self.age_start
                if worker_accept == 1:
                    action = 1
                    # compute total discounted reward 
                    t = self.current_age - self.age_start  # time step when job is accepted
                    T = self.age_retire - self.age_start  # total time steps until retirement

                    # new reward func
                    if self.gamma != 1:
                        reward_raw = wage_offer * (1 - self.gamma**(T - t)) / (1 - self.gamma)
                    else:
                        reward_raw = wage_offer * (T - t)
                    # new feature: age scaling
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
            
            # i dont like this, find a way to make it better
            # firm-worker interaction
            # worker only mode?? figure oujt why this is
            # firm-wroker interaciotn??
            else:
                t = self.current_age - self.age_start  # time step when job is accepted
                T = self.age_retire - self.age_start
                # reservation_wage = self.res_wage_table[self.current_age]
                if worker_accept == 1:
                    # compute total discounted reward 
                    # t = self.current_age - self.age_start  # time step when job is accepted
                    # T = self.age_retire - self.age_start  # total time steps until retirement

                    # new reward func
                    if self.gamma != 1:
                        worker_reward_raw = wage_offer * (1 - self.gamma**(T - t)) / (1 - self.gamma)
                    else:
                        worker_reward_raw = wage_offer * (T - t)

                    firm_reward = self.value_match - wage_offer

                    # new feature: age scaling
                    age_scaling = 0.5 + 0.5 * ((self.current_age - self.age_start) / (self.age_retire - self.age_start))
                    worker_reward = worker_reward_raw * age_scaling

                    self.done = True
                    action = 1

                else:
                    firm_reward = -self.unemployment_penalty
                    cont_value = self.res_wage_table[self.current_age] * (1 - self.gamma**(T - t)) / (1 - self.gamma)
                    worker_reward = -self.unemployment_penalty + cont_value
                    self.done = False
                    action = 0

                if not self.done:
                    self.current_age += 1
                    if self.current_age >= self.age_retire:
                        self.done = True

                return firm_reward, worker_reward, self.done, action
        
        elif self.worker_only_mode:
            if wage_offer >= self.reservation_wage:
                action = 1
                self.done = True
                reward = wage_offer
                # print("Worker accepted offer of ", wage_offer)
            else:
                action = 0
                self.done = False
                reward = 0
                # print("Worker rejected offer of ", wage_offer)

            return reward, self.done, action # finish step here
        
        elif self.firm_only_mode:
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

        elif self.random_mode:
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
        
        else:
            raise ValueError("Invalid mode")
    
    # add code in simulations and reset/step jugak
    def calculate_reservation_wage(self, gamma, wage_min, wage_max, unemployment_penalty, num_wages=100):
        # calc the dynamic reservation wage based on the McCall model
        wage_range = np.linspace(wage_min, wage_max, num_wages)
        value_function = np.zeros(num_wages)  # Value function initialization
        tolerance = 1e-3
        max_iterations = 1000

        for age in range(max_iterations):
            new_value_function = np.zeros_like(value_function)
            for i, wage in enumerate(wage_range):
                accept_value = wage  # Value of accepting the wage
                reject_value = -unemployment_penalty + gamma * np.mean(value_function)  # Value of rejecting
                new_value_function[i] = max(accept_value, reject_value)

            if np.max(np.abs(new_value_function - value_function)) < tolerance:
                break

            value_function = new_value_function

        # The reservation wage is where accepting equals rejecting
        reservation_wage = wage_range[np.argmax(value_function >= (-unemployment_penalty + gamma * np.mean(value_function)))]
        return reservation_wage
    
    # is this needed?
    def age_based_reservation_wage(self, mode):
        """
        Precompute the reservation wage for each age from age_start to age_retire - 1.
        Returns a dictionary mapping age to reservation wage.
        """
        res_table = {}
        wage_values = np.linspace(self.wage_min, self.wage_max, 101)
        p = np.ones_like(wage_values) / len(wage_values)
        beta = 0.1 # smoothing parameter
        
        for age in range(self.age_start, self.age_retire):
            T = self.age_retire - age
            V = np.zeros(T+1)
            for t in range(T-1, -1, -1):
                wait_value = -self.unemployment_penalty + self.gamma * V[t+1]
                if mode == "worker":

                    # new reward func
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
            # Define the reservation wage at this age (using the indifference condition)
            # res_table[age] = wait_value if T >= 1 else 0.0
            
            valid_indices = np.where(accept_reward >= wait_value)[0]
            if len(valid_indices) > 0:
                res_wage_for_age = wage_values[valid_indices[0]]
            else:
                # If none of them exceed wait_value, then the reservation wage might be the max wage
                res_wage_for_age = wage_values[-1]

            res_table[age] = res_wage_for_age
        
        return res_table

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
        # Discretize wage offer
        wage_bins = np.linspace(env.wage_min, env.wage_max, self.n_states + 1)
        wage_state = np.clip(np.digitize(wage_offer, wage_bins) - 1, 0, self.n_states - 1)

        if self.use_age_based and worker_age is not None:
            # Discretize age into two bins: "young" and "old" for simplicity
            age_threshold = (env.age_retire - env.age_start) / 2
            age_state = 0 if worker_age < (env.age_start + age_threshold) else 1
            # Combine wage state and age state into a single state
            combined_state = wage_state * 2 + age_state
            return np.clip(combined_state, 0, len(self.q_table) - 1)
        else:
            # Only wage-based state
            return wage_state

    def choose_action(self, state, episode, n_episodes):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

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
        # epsilon = max(0.01, self.epsilon * (1 - episode / n_episodes) ** 0.5) # less exploration
        epsilon = max(0.1, self.epsilon * (1 - episode / n_episodes) * 0.5) # more exploration
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
        
        ### prev version
        # else:
        #     best_next_action = np.argmax(self.q_table[next_state])
        #     td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        # td_error = td_target - self.q_table[state, action]
        # self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

# %%
def simulate(env, n_episodes, agent, evaluate_after=20, timesteps_per_sim=100):
    if env.age_based_mode:
        if env.worker_only_mode:
            # Here we use our DQNWorkerAgent (the worker learns via PyTorch)
            worker = agent 
            num_age_steps = env.age_retire - env.age_start
            age_offer_data = []
            acceptance_data = []
            # For plotting, we record the environment’s computed reservation wage per age
            recorded_rw = np.zeros((n_episodes, num_age_steps))
            
            for episode in range(n_episodes):
                env.current_age = env.reset()  # env.reset() returns a random starting age in [age_start, age_retire)
                done = False
                age_index = env.current_age - env.age_start
                step_count = 0

                while not done:
                    # Firm offers a wage from a Gaussian (firms do not learn)
                    base = env.res_wage_table[env.current_age]
                    wage_offer = np.random.normal(loc=base, scale=(env.wage_max - env.wage_min)*0.1)
                    wage_offer = np.clip(wage_offer, env.wage_min, env.wage_max)
                    

                    
                    # old
                    # state = [wage_offer / env.wage_max,
                            # (env.current_age - env.age_start) / (env.age_retire - env.age_start)]
                    
                    # new
                    remaining = ((env.age_retire - env.current_age) / (env.age_retire - env.age_start))
                    state = [wage_offer / env.wage_max,
                            ((env.current_age - env.age_start) / (env.age_retire - env.age_start)),
                            remaining]


                    # Agent selects action: 0 = reject, 1 = accept
                    action = worker.select_action(state)
                    
                    # Step the environment with the current wage offer
                    worker_reward, done, env_action = env.step(wage_offer, worker_accept=action)
                    age_offer_data.append((env.current_age, wage_offer))
                    acceptance_data.append(env_action)

                    # Build the next state.
                    if not done:
                        new_age = env.current_age
                        new_remaining = ((env.age_retire - new_age) / (env.age_retire - env.age_start))
                        next_state = [wage_offer / env.wage_max,
                                    ((new_age - env.age_start) / (env.age_retire - env.age_start)),
                                    new_remaining]
                    else:
                        next_state = [0.0, 0.0, 0.0]
                    
                    # Store the transition in the agent’s memory.
                    worker.store_transition(state, action, worker_reward, next_state, done)
                    worker.train_step()
                    
                    # step_count += 1
                    # if step_count % 4 == 0:
                    #     worker.train_step()
                    
                    # Record the environment’s reservation wage at the current age
                    if env.current_age < env.age_retire:
                        recorded_rw[episode, age_index] = env.res_wage_table[env.current_age]
                    else:
                        # If current_age reached or exceeded age_retire, mark as done.
                        done = True

                    if not done:
                        env.current_age = new_age
                        if env.current_age < env.age_retire:
                            age_index =env.current_age - env.age_start
                        else:
                            done = True
                            
            # Return the recorded reservation wages for plotting.
            return recorded_rw, age_offer_data, acceptance_data
        
        elif env.firm_only_mode:
            # Here 'agent' is a DQNFirmAgent instance.
            firm = agent  
            total_offers = []    # Record all wage offers made
            eval_acceptances = []  # Record wage offer indices that lead to acceptance
            age_offer_data = []  # Record the age and wage offer for each episode
            
            # Run simulation over episodes.
            for sim in range(n_episodes):
                env.reset()  # In age-based mode, reset() sets current_age = age_start and precomputes reservation wage table
                done = False
                # In age-based mode, the state for the firm is the worker's normalized age.
                state = [(env.current_age - env.age_start) / (env.age_retire - env.age_start)]
                
                for t in range(timesteps_per_sim):
                    if done:
                        break
                        
                    # Firm selects an action using its DQN (action index in {0,..., n_wage_levels-1})
                    action_index = firm.select_action(state)
                    
                    # Map the action index to a wage offer.
                    # For example, if there are n_wage_levels discrete wage offers:
                    wage_offer = env.wage_min + action_index * (env.wage_max - env.wage_min) / (firm.n_wage_levels - 1)
                    total_offers.append(wage_offer)

                    age_offer_data.append((env.current_age, wage_offer))
                    
                    # Step the environment in age-based mode.
                    # In age-based mode, env.step(wage_offer) returns (firm_profit, worker_reward, done, acceptance)
                    firm_reward, done, acceptance = env.step(wage_offer)

                    # Define next state: update the worker's normalized age.
                    next_state = ([(env.current_age - env.age_start) / (env.age_retire - env.age_start)]
                                if not done else [0.0])
                    d = 1.0 if done else 0.0
                    
                    # Store the transition and perform training.
                    firm.store_transition(state, action_index, firm_reward, next_state, d)
                    firm.train_step()
                    
                    # Record acceptance if worker accepts.
                    if acceptance == 1:
                        eval_acceptances.append(action_index)
                        break
                    
                    # Update state for next timestep.
                    state = next_state
                
            return total_offers, eval_acceptances, firm.q_network, age_offer_data

    elif env.worker_only_mode:
        worker = agent
        eval_rewards = []
        eval_acceptances = []

        all_wage_offers = []
        all_acceptances = []

        for sim in range(evaluate_after + 1):
            env.reset()
            cumulative_reward = 0
            acceptances = []

            wage_offers = np.linspace(env.wage_min, env.wage_max, 100)
            np.random.shuffle(wage_offers)

            all_wage_offers.extend(wage_offers)

            for i in range(timesteps_per_sim):
                wage_offer = wage_offers[i]
                state = worker.get_state(wage_offer, env)

                # Enforce acceptance above reservation wage during evaluation
                if sim >= evaluate_after and wage_offer >= env.reservation_wage:
                    action_taken = 1
                    # print("Worker accepted offer of ", wage_offer)
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

            if sim == evaluate_after:
                eval_rewards.append(cumulative_reward)
                eval_acceptances.extend(acceptances)

        # Process evaluation data only
        num_bins = 10
        wage_bins = np.linspace(env.wage_min, env.wage_max, num_bins + 1)
        wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

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

        # Return evaluation-only results
        return eval_rewards, eval_acceptance_rates, worker.q_table

    elif env.firm_only_mode:
        firm = agent
        eval_acceptances = []  # Track acceptances during evaluation
        total_offers = []      # Track all offers

        for sim in range(evaluate_after + 1):  # Split into training and evaluation
            env.reset()
            firm_offers = np.zeros(firm.n_wage_levels)  # Track total offers by firm
            accepted_firm_offers = np.zeros_like(firm_offers)

            for t in range(timesteps_per_sim):
                state = 0  # Firm logic uses a single state for simplicity
                action_index = firm.choose_action(state, sim, n_episodes)

                # Convert action index to a wage offer
                wage_offer = env.wage_min + action_index * (env.wage_max - env.wage_min) / firm.n_wage_levels
                total_offers.append(wage_offer)

                # Step in environment
                firm_reward, done, action_taken = env.step(wage_offer)

                if sim < evaluate_after:  # Training phase
                    firm.update_q_table(state, action_index, firm_reward, state, done)

                if action_taken == 1:  # Worker accepted the offer
                    accepted_firm_offers[action_index] += 1

                if done:  # Stop if worker exits
                    break

            if sim >= evaluate_after:  # Record evaluation data
                eval_acceptances.extend(accepted_firm_offers)

        # Process evaluation data for plotting
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

        # Return evaluation results
        return total_offers, eval_acceptance_rates, firm.q_table

    # firm and worker interaction
    elif env.random_mode:
        firm = FirmAgent(n_states=40, n_wage_levels=20, epsilon=1.0, epsilon_decay=0.995)
        worker = WorkerAgent(n_states=150, n_actions=2, use_age_based=False, epsilon=1.0, epsilon_decay=0.995)

        firm_offers = np.zeros(firm.n_wage_levels)
        worker_accepts = np.zeros_like(firm_offers)
        firm_profits = []

        for episode in range(n_episodes):
            env.reset()
            done = False
            total_profit = 0

            while not done:
                # Firm state logic
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
                    worker_reward = -1  # Penalty for rejecting
                    firm_reward = -env.unemployment_penalty  # Firm penalty for rejection
                    action_taken = 0
                    if np.random.rand() < env.p_exit:  # Worker exits
                        done = True
                
                firm_reward = env.value_match - wage_offer if action_taken == 1 else -env.unemployment_penalty
                worker_reward = wage_offer if action_taken == 1 else -env.unemployment_penalty

                next_firm_state = firm_state
                next_worker_state = worker_state

                firm.update_q_table(firm_state, firm_action_index, firm_reward, next_firm_state, done)
                worker.update_q_table(worker_state, worker_action, worker_reward, next_worker_state, done)

                if action_taken == 1:
                    worker_accepts[firm_action_index] += 1
                    total_profit += firm_reward
            
            firm_profits.append(total_profit)
        
        acceptance_rates = np.divide(worker_accepts, firm_offers, out=np.zeros_like(worker_accepts, dtype=float), where=firm_offers != 0)

        return firm_offers, worker_accepts, acceptance_rates, firm_profits 

    else:
        raise ValueError("Invalid mode")

# %%
def compute_learned_res_wage(worker_agent, env, wage_points=100):
    """
    Evaluate the worker's trained DQN policy by checking the lowest wage they accept at each age.
    """
    ages = np.arange(env.age_start, env.age_retire)
    wages = np.linspace(env.wage_min, env.wage_max, wage_points)
    res_wages = []
    
    # Sample more detailed diagnostics
    diagnostic_ages = [20, 50, 100, 140]
    q_values_by_age = {age: [] for age in diagnostic_ages}

    for age in ages:
        found_res_wage = env.wage_max
        found_accept = False
        
        for w in wages:
            remaining = ((env.age_retire - age) / (env.age_retire - env.age_start)) 
            state = [w / env.wage_max, (age - env.age_start) / (env.age_retire - env.age_start), remaining]
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = worker_agent.q_network(state_t)
            action = torch.argmax(q_values, dim=1).item()
            
            # Record Q-values for specific ages
            if age in diagnostic_ages:
                q_values_by_age[age].append((w, q_values.detach().numpy()))
            
            if action == 1:  # Accept
                found_res_wage = w
                found_accept = True
                break
        
        res_wages.append(found_res_wage)

    # Print detailed diagnostics
    for age in diagnostic_ages:
        print(f"\nQ-values at age {age}:")
        for wage, q_vals in q_values_by_age[age][::10]:  # Print every 10th value
            print(f"  Wage {wage:.2f}: Reject={q_vals[0][0]:.2f}, Accept={q_vals[0][1]:.2f}")
        
    return ages, res_wages

# %%
def simulate_age_based(env, n_episodes, firm_agent, worker_agent, timesteps=100):
    age_wage_data = []
    acceptance_data = []

    for episode in range(n_episodes):
        env.current_age = env.reset()
        done = False

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

    return age_wage_data, acceptance_data

# %%
def evaluate(env, agent, mode, timesteps, simulations):
    # Reset environment and agent state
    env.reset()
    
    eval_wage_offers = []
    eval_acceptances = []

    for sim in range(simulations):
        wage_offers = np.linspace(env.wage_min, env.wage_max, timesteps)
        np.random.shuffle(wage_offers)

        for i in range(timesteps):
            wage_offer = wage_offers[i]

            if mode == "worker":
                # Worker evaluates offer using learned policy
                action = 1 if wage_offer >= env.reservation_wage else 0
            elif mode == "firm":
                # Firm chooses a wage to offer based on policy
                state = 0  # Firm has one state in base case
                action = agent.choose_action(state, sim, simulations)
                wage_offer = env.wage_min + action * (env.wage_max - env.wage_min) / agent.n_wage_levels

                # Worker decides whether to accept the offer
                action = 1 if wage_offer >= env.reservation_wage else 0
            else:
                raise ValueError(f"Invalid mode: {mode}")

            # Record the results
            eval_wage_offers.append(wage_offer)
            eval_acceptances.append(action)

            if action == 1:  # Stop when an offer is accepted
                break

    return eval_wage_offers, eval_acceptances


# %%
def evaluate_policy_over_ages(worker_agent, env, wage_points=100):
    """
    Evaluate the worker's trained DQN policy by checking the lowest wage they accept at each age.
    """
    ages = np.arange(env.age_start, env.age_retire)
    reservation_wages = []
    
    # Create a grid of possible wage offers.
    wage_grid = np.linspace(1, env.wage_max, wage_points)
    print(f"wage_grid: {wage_grid}")
    
    for age in ages:
        found_reservation = env.wage_max  # default to max if no acceptance is found
        # Calculate normalized age and remaining time once for the current age.
        # rescaled for better variation
        norm_age = ((age - env.age_start) / (env.age_retire - env.age_start))
        remaining = ((env.age_retire - age) / (env.age_retire - env.age_start))
        
        for wage in wage_grid:
            norm_wage = wage / env.wage_max
            # Now the state has 3 elements: normalized wage, normalized age, and remaining time.
            state = [norm_wage, norm_age, remaining]
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = worker_agent.q_network(state_t)
            action = torch.argmax(q_values, dim=1).item()

            # Debugging print
            print(f"Age {age}, Wage {wage:.2f}, Q-values: {q_values.detach().numpy()}, Action: {action}")
            
            if action == 1:  # Accept
                found_reservation = wage
                break
        
        reservation_wages.append(found_reservation)
    return ages, reservation_wages


# %%
def normalize_age(age_start, age_retire, age_wage_data, portion = 0.2):
    # splits into episodes
    episodes = []
    current_ep = []
    for (age, wage) in age_wage_data:
        # Start a new episode if age == age_start and there is an existing episode.
        if age == age_start and current_ep:
            episodes.append(current_ep)
            current_ep = [(age, wage)]
        else:
            current_ep.append((age, wage))
    if current_ep:
        episodes.append(current_ep)
    
    # Determine how many episodes to use (e.g., last 20% episodes)
    num_eps = len(episodes)
    start_index = int(num_eps * (1 - portion))
    filtered_eps = episodes[start_index:]
    
    # Flatten the list back to a single list of (age, wage)
    filtered_data = [entry for ep in filtered_eps for entry in ep]

    age_span = age_retire - age_start
    sum_wages = np.zeros(age_span)
    count_wages = np.zeros(age_span)

    for (age, wage) in filtered_data:
        idx = age - age_start
        if 0 <= idx < age_span:
            sum_wages[idx] += wage
            count_wages[idx] += 1

    avg_wages = sum_wages / np.maximum(count_wages, 1)
    ages = np.arange(age_start, age_retire)
    return ages, avg_wages

# %%
# Define a helper function to extract the accepted wages for each age.
def extract_accepted_wages(age_wage_data, acceptance_data):
    
    accepted = {}
    for (age, wage), act in zip(age_wage_data, acceptance_data):
        # We consider a wage only if the worker accepted (act == 1).
        if act == 1:
            if age in accepted:
                accepted[age].append(wage)
            else:
                accepted[age] = [wage]
    return accepted

# %%
# For each case, create a sorted list of ages and the corresponding average accepted wage.
def average_by_age(accepted_dict):
    ages = sorted(accepted_dict.keys())
    avg_wages = [np.mean(accepted_dict[age]) for age in ages]
    return ages, avg_wages

# %%
# Number of episodes
n_episodes = 21000

# %%
# --- Simulate age-based environment with different gamma values ---
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
    n_wage_levels=50,
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
    n_wage_levels=50,
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
age_wage_data_low, acceptance_data_low = simulate_age_based(
    env_age_low, 
    n_episodes, 
    firm_agent_low, 
    worker_agent_low
    )

age_wage_data_high, acceptance_data_high = simulate_age_based(
    env_age_high, 
    n_episodes, 
    firm_agent_high, 
    worker_agent_high
    )
# normalize ages for plot
norm_ages_low, avg_offers_by_age_low = normalize_age(
    env_age_low.age_start, 
    env_age_low.age_retire, 
    age_wage_data_low
    )

norm_ages_high, avg_offers_by_age_high = normalize_age(
    env_age_high.age_start, 
    env_age_high.age_retire, 
    age_wage_data_high
    )

# %%
# age based worker only but with evaluation - interaction mode
# eval_ages_low, res_wages_low = evaluate_policy_over_ages(worker_agent_low, env_age_low, wage_points=100)
# eval_ages_high, res_wages_high = evaluate_policy_over_ages(worker_agent_high, env_age_high, wage_points=100)

eval_ages_low,  res_wages_low  = compute_learned_res_wage(worker_agent_low,  env_age_low)
eval_ages_high, res_wages_high = compute_learned_res_wage(worker_agent_high, env_age_high)

plt.figure(figsize=(8,5))
plt.plot(eval_ages_low, res_wages_low, marker='o', label='Low Gamma (0.01)')
plt.plot(eval_ages_high, res_wages_high, marker='o', label='High Gamma (0.9999)')
plt.xlabel("Worker Age")
plt.ylabel("Reservation Wage")
plt.title("Worker Reservation Wage vs. Age (Interaction)")
plt.legend()
plt.show()

# %%
# firm side in age-based firm-worker interaction

plt.figure(figsize=(7,5))
plt.plot(norm_ages_low, avg_offers_by_age_low, marker='o', label='Low Gamma (0.01)')
plt.plot(norm_ages_high, avg_offers_by_age_high, marker='o', label='High Gamma (0.9999)')
plt.xlabel("Worker Age")
plt.ylabel("Average Firm Wage Offer")
plt.title("Firm Wage Offers vs. Age (Interaction)")
plt.legend()
plt.show()

# %%
# -------------------- AGE-BASED WORKER-ONLY MODE --------------------

# age-based: low discount factor
age_based_env_low = McCallModelEnv(age_based_mode=True, worker_only_mode=True, firm_only_mode=False, gamma=0.01)
worker_low = DQNWorkerAgent(state_dim=3, action_dim=2, gamma=0.01)
results_low, age_offer_data_low, acceptance_data_low = simulate(age_based_env_low, n_episodes, agent=worker_low)

# age-based: high discount factor
age_based_env_high = McCallModelEnv(age_based_mode=True, worker_only_mode=True, firm_only_mode=False, gamma=0.9999)
worker_high = DQNWorkerAgent(state_dim=3, action_dim=2, gamma=0.9999)
results_high, age_offer_data_high, acceptance_data_high = simulate(age_based_env_high, n_episodes, agent=worker_high)

ages_low, avg_offers_by_age_low = normalize_age(age_based_env_low.age_start, age_based_env_low.age_retire, age_offer_data_low)
ages_high, avg_offers_by_age_high = normalize_age(age_based_env_high.age_start, age_based_env_high.age_retire, age_offer_data_high)


# %%
# age based worker only but with evaluation
ages_low, res_wages_low = evaluate_policy_over_ages(worker_low, age_based_env_low, wage_points=100)
ages_high, res_wages_high = evaluate_policy_over_ages(worker_high, age_based_env_high, wage_points=100)

plt.figure(figsize=(8,5))
plt.plot(ages_low, res_wages_low, marker='o', label='Low Gamma (0.01)')
plt.plot(ages_high, res_wages_high, marker='o', label='High Gamma (0.999)')
plt.xlabel("Worker Age")
plt.ylabel("Reservation Wage")
plt.title("Worker Reservation Wage vs. Age")
plt.legend()
plt.show()

# %%
# ---- age-based firm only mode ----

env_low = McCallModelEnv(age_based_mode=True, firm_only_mode=True, gamma=0.01)

env_high = McCallModelEnv(age_based_mode=True, firm_only_mode=True, gamma=0.9999)

# Create a DQN firm agent for each. (Set its gamma to 1.0 to avoid double discounting.)
firm_agent_low = DQNFirmAgent(state_dim=1, n_wage_levels=100, gamma=0.01, lr=1e-3)
firm_agent_high = DQNFirmAgent(state_dim=1, n_wage_levels=100, gamma=0.9999, lr=1e-3)

# simulations
offers_low, acceptances_low, q_net_low, age_offer_data_low = simulate(env_low, n_episodes, firm_agent_low)
offers_high, acceptances_high, q_net_high, age_offer_data_high = simulate(env_high, n_episodes, firm_agent_high)

# normalize ages for plot
ages_low, avg_offers_by_age_low = normalize_age(env_low.age_start, env_low.age_retire, age_offer_data_low)
ages_high, avg_offers_by_age_high = normalize_age(env_high.age_start, env_high.age_retire, age_offer_data_high)

plt.plot(ages_low, avg_offers_by_age_low, marker='o', label='Low Gamma (0.01)')
plt.plot(ages_high, avg_offers_by_age_high, marker='o', label='High Gamma (0.9999)')
plt.xlabel('Worker Age')
plt.ylabel('Average Wage Offer')
plt.title('Firm Wage Offers vs. Age')
plt.legend()
plt.show()


# %%
# Environment setup
worker_only_env = McCallModelEnv(worker_only_mode=True)
firm_only_env = McCallModelEnv(firm_only_mode=True, p_exit=0.02)
random_env = McCallModelEnv(random_mode=True, p_exit=0.02) # firm + worker but need to fix a bit (add worker)

# Agent setup
firm = FirmAgent(n_states=40, n_wage_levels=20)
worker = WorkerAgent(n_states=150, n_actions=2, use_age_based=False)

# -- simulations --

# worker only
wage_bins, eval_acceptance_rates, worker_q_table = simulate(worker_only_env, n_episodes, agent=worker)
worker_eval_offers, worker_eval_acceptances = evaluate(
    worker_only_env, worker, mode="worker", timesteps=100, simulations=50
)

# firm only
total_offers, acceptance_rates, firm_q_table = simulate(firm_only_env, n_episodes, firm)
firm_eval_offers, firm_eval_acceptances = evaluate(
    firm_only_env, firm, mode="firm", timesteps=100, simulations=60
)

# mixed firm + worker
firm_offers, worker_accepts, acceptance_rates, firm_profits = simulate(random_env, n_episodes, agent=firm)


# %%
# Define finer bins with better alignment
num_bins = 50
wage_bins = np.linspace(worker_only_env.wage_min, worker_only_env.wage_max + 1e-5, num_bins + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

# Process acceptance data
acceptance_counts = np.zeros(num_bins)
offer_counts = np.zeros_like(acceptance_counts)

for offer, accept in zip(worker_eval_offers, worker_eval_acceptances):
    bin_idx = np.digitize(offer, wage_bins) - 1
    if 0 <= bin_idx < num_bins:
        offer_counts[bin_idx] += 1
        acceptance_counts[bin_idx] += accept

# Calculate acceptance rates
eval_acceptance_rates = np.divide(
    acceptance_counts,
    offer_counts,
    out=np.zeros_like(acceptance_counts, dtype=float),
    where=offer_counts != 0
)

# Debugging outputs
# print(f"Wage Bin Midpoints: {wage_bin_midpoints}")
# print(f"Eval Acceptance Rates: {eval_acceptance_rates}")

# Plot with thinner bars
plt.figure(figsize=(6, 4))
bar_width = (wage_bins[1] - wage_bins[0]) * 0.4  # Adjust for thinner bars
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


# %%
# Define finer bins with better alignment
num_bins = 50
wage_bins = np.linspace(firm_only_env.wage_min, firm_only_env.wage_max + 1e-5, num_bins + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

# Process firm wage offer frequency data
offer_counts = np.zeros(num_bins)

for offer in total_offers:  # Assuming total_offers contains all wage offers from simulations
    bin_idx = np.digitize(offer, wage_bins) - 1
    if 0 <= bin_idx < num_bins:
        offer_counts[bin_idx] += 1

# Plot the firm offer frequency
plt.figure(figsize=(6, 4))
bar_width = (wage_bins[1] - wage_bins[0]) * 0.8  # Adjust bar width for clarity
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


# %%
# Calculate midpoints for wage bins
wage_bins = np.linspace(random_env.wage_min, random_env.wage_max, firm.n_wage_levels + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

# Plot firm offers
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(wage_bin_midpoints, firm_offers, color='blue', alpha=0.7, label='Total Firm Offers')
ax.set_title('Firm Wage Offer Frequency')
ax.set_xlabel('Wage Offered')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.show()

# Plot acceptance rates
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(wage_bin_midpoints, acceptance_rates, color='green', alpha=0.7, label='Worker Acceptance Rates')
ax.set_title('Worker Acceptance Rates')
ax.set_xlabel('Wage Offered')
ax.set_ylabel('Acceptance Rate')
ax.legend()
plt.tight_layout()
plt.show()



