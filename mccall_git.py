# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# defining the environment
class McCallModelEnv:
    def __init__(self, wage_min=0, wage_max=100, value_match=200, age_start=20, age_retire=60,
                  unemployment_penalty=5.0, gamma=1., p_exit=0.01, age_based_mode=False, 
                  use_random_reservation_wage=False, worker_only_mode=False, firm_only_mode=False, random_mode = False):
        
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
        self.use_random_reservation_wage = use_random_reservation_wage

        # worker/firm only mode
        self.worker_only_mode = worker_only_mode
        self.firm_only_mode = firm_only_mode
        self.random_mode = random_mode

        self.gamma = gamma
        self.reset()

    def reset(self):
        self.done = False
        if self.worker_only_mode or self.firm_only_mode:
            self.reservation_wage = 50
            return self.reservation_wage
        elif self.random_mode:
            self.reservation_wage = np.random.uniform(
                self.wage_min + 0.3 * (self.wage_max - self.wage_min),
                self.wage_min + 0.7 * (self.wage_max - self.wage_min)
            )
        elif self.age_based_mode:
            self.current_age = np.random.randint(self.age_start, self.age_retire)
            return self.current_age
    
    def step(self, wage_offer):
        if self.worker_only_mode:
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
        
        # if age-based
        elif self.age_based_mode:
            # Age-based reservation wage
            reservation_wage = np.clip(
                self.wage_min + (self.wage_max - self.wage_min) * 
                (self.age_retire - self.current_age) / (self.age_retire - self.age_start),
                self.wage_min, self.wage_max
            )
            if wage_offer >= reservation_wage:
                action = 1 # accept
                firm_profit = self.value_match - wage_offer
                worker_reward = wage_offer
                self.done = True
            else:
                action = 0 # reject
                firm_profit = -self.unemployment_penalty
                worker_reward = -self.unemployment_penalty
                self.done = False

            self.current_age += 1
            if self.current_age >= self.age_retire:
                self.done = True

            return firm_profit, worker_reward, self.done, action
        
        else:
            raise ValueError("Invalid mode")
        
        

# %%
class WorkerAgent:
    def __init__(self, n_states, n_actions, use_age_based=False, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
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

class FirmAgent:
    def __init__(self, n_states, n_wage_levels=10, alpha=0.1, gamma=1.0, epsilon=0.5, epsilon_decay=0.999):
        self.q_table = np.zeros((n_states, n_wage_levels))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.n_wage_levels = n_wage_levels
    
    def discretize_action(self, wage, wage_min, wage_max):
        return min(int((wage - wage_min) / (wage_max - wage_min) * self.n_wage_levels), self.n_wage_levels - 1)

    def choose_action(self, state, episode, n_episodes):
        epsilon = max(0.01, self.epsilon * (1 - episode / n_episodes)**0.5)
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_wage_levels)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

# %%
def simulate(env, n_episodes, agent):
    if env.worker_only_mode:
        worker = agent
        wage_offers = []
        acceptances = []

        # check how to do RL for worker only mode - let worker determine reservation wage

        for episode in range(n_episodes):
            env.reset()  # Reset environment (new reservation wage)
            episode_wage_offers = np.linspace(env.wage_min, env.wage_max, 20)
            np.random.shuffle(episode_wage_offers)

            for wage_offer in episode_wage_offers:
                # Get current state
                state = worker.get_state(wage_offer, env)

                # Choose an action (0 = Reject, 1 = Accept)
                action = worker.choose_action(state, episode, n_episodes)

                # Step in the environment
                _, done, action_taken = env.step(wage_offer if action == 1 else 0)
                wage_offers.append(wage_offer)
                acceptances.append(action_taken)

                # Compute reward and next state
                # reward = wage_offer if action_taken == 1 else -1  # Penalty for rejection

                # testing other method of reward calc
                reward = wage_offer if action_taken == 1 and wage_offer >= env.reservation_wage else -1

                next_state = worker.get_state(wage_offer, env)

                # Update Q-table
                worker.update_q_table(state, action, reward, next_state, done)

                if action_taken == 1:  # Stop after acceptance
                    break

            # Decay epsilon
            worker.decay_epsilon()

        # Bin wage offers and calculate acceptance rates
        wage_bins = np.linspace(env.wage_min, env.wage_max, 20)
        bin_indices = np.digitize(wage_offers, wage_bins, right=True)
        acceptance_rates = np.zeros(len(wage_bins) - 1)

        for i in range(1, len(wage_bins)):
            bin_acceptances = [acceptances[j] for j in range(len(bin_indices)) if bin_indices[j] == i]
            acceptance_rates[i - 1] = np.mean(bin_acceptances) if bin_acceptances else 0

        return wage_bins, acceptance_rates, worker.q_table
    
    elif env.firm_only_mode:
        firm = agent
        total_firm_offers = np.zeros(firm.n_wage_levels)  # Track total offers by firm
        accepted_firm_offers = np.zeros_like(total_firm_offers)  # Track accepted offers
        acceptance_rates = np.zeros_like(total_firm_offers)

        for episode in range(n_episodes):
            env.reset()  # Reset environment for each episode
            done = False

            while not done:
                # Firm state logic (simplified for base case)
                state = 0  # Single state since this is the base case

                # Firm chooses a wage offer (action) based on its policy
                firm_action_index = firm.choose_action(state, episode, n_episodes)
                firm_action = env.wage_min + firm_action_index * (env.wage_max - env.wage_min) / firm.n_wage_levels

                # Record the wage offer
                total_firm_offers[firm_action_index] += 1

                # Environment step
                firm_reward, done, action_taken = env.step(firm_action)

                # Track accepted offers
                if action_taken == 1:  # Worker accepted
                    accepted_firm_offers[firm_action_index] += 1

                # Update Q-table
                firm.update_q_table(state, firm_action_index, firm_reward, state, done)

                if done:  # Stop if the worker exits or accepts
                    break

        # Calculate acceptance rates
        acceptance_rates = np.divide(
            accepted_firm_offers,
            total_firm_offers,
            out=np.zeros_like(total_firm_offers, dtype=float),
            where=total_firm_offers != 0
        )

        return total_firm_offers, accepted_firm_offers, acceptance_rates
    
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

    elif env.age_based_mode:
        firm = agent
        # Track firm decisions and worker responses
        firm_offers = np.zeros((firm.n_wage_levels, env.age_retire - env.age_start)) if not env.use_random_reservation_wage else np.zeros((firm.n_wage_levels))
        worker_accepts = np.zeros_like(firm_offers)
        firm_profits = []

        for episode in range(n_episodes):
            worker_age = env.reset()
            done = False
            total_profit = 0

            while not done:

                # firm state logic
                if env.use_random_reservation_wage:
                    firm_state = 0 # one state for all workers
                else:
                    # firm's state is based on normalized worker age
                    firm_state = int((worker_age - env.age_start) / (env.age_retire - env.age_start) * firm.q_table.shape[0])
                    firm_state = min(firm_state, firm.q_table.shape[0] - 1)  # Prevent out-of-bounds error

                # Firm chooses a wage to offer
                firm_action_index = firm.choose_action(firm_state, episode, n_episodes)
                firm_action = env.wage_min + firm_action_index * (env.wage_max - env.wage_min) / firm.n_wage_levels

                # Environment step
                firm_reward, worker_reward, done, action = env.step(firm_action)
                total_profit += firm_reward

                if env.use_random_reservation_wage:
                    firm_offers[firm_action_index] += 1
                    # worker accepted
                    if action == 1: 
                        worker_accepts[firm_action_index] += 1
                else:
                    age_index = worker_age - env.age_start
                    firm_offers[firm_action_index, age_index] += 1
                    if action == 1:  # Worker accepted
                        worker_accepts[firm_action_index, age_index] += 1

                # Update Q-table
                firm.update_q_table(firm_state, firm_action_index, firm_reward, firm_state, done)

                # If worker accepts, end the episode
                if done:
                    break
            
            firm_profits.append(total_profit)

        acceptance_rates = np.divide(worker_accepts, firm_offers, out=np.zeros_like(worker_accepts, dtype=float), where=firm_offers != 0)

        return firm_offers, worker_accepts, acceptance_rates, firm_profits
    
    else:
        raise ValueError("Invalid mode")

# %%
# Environment setup
worker_only_env = McCallModelEnv(worker_only_mode=True)
firm_only_env = McCallModelEnv(firm_only_mode=True, p_exit=0.02)
random_env = McCallModelEnv(random_mode=True, p_exit=0.02) # firm + worker but need to fix a bit (add worker)
age_based_env = McCallModelEnv(age_based_mode=True)

# Agent setup
firm = FirmAgent(n_states=40, n_wage_levels=20)
worker = WorkerAgent(n_states=150, n_actions=2, use_age_based=False)

# Number of episodes
n_episodes = 150000

# -- simulations --

# worker only
wage_bins, worker_only_acceptance_rates, worker_q_table = simulate(worker_only_env, n_episodes, agent=worker)

# firm only
total_offers, accepted_offers, acceptance_rates = simulate(firm_only_env, n_episodes, firm)

# mixed firm + worker
firm_offers, worker_accepts, acceptance_rates, firm_profits = simulate(random_env, n_episodes, agent=firm)

age_based_results = simulate(age_based_env, n_episodes, agent=firm)

# %%
# Calculate midpoints of the wage bins for plotting
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

# Plotting
fig, ax = plt.subplots(figsize=(6, 4))

# Bar plot
ax.bar(wage_bin_midpoints, worker_only_acceptance_rates, width=5.0, color='blue', alpha=1.0, label='Acceptance Rates')

# Add the reservation wage line
reservation_wage = worker_only_env.reservation_wage
ax.axvline(reservation_wage, color='green', linestyle='--', linewidth=2, label='Reservation Wage')

# Set titles and labels
ax.set_title("Worker Acceptance Rates")
ax.set_xlabel("Wage Offered")
ax.set_ylabel("Acceptance Rate")
ax.legend()

plt.tight_layout()
plt.show()


# %%
# firm only env
wage_bins = np.linspace(random_env.wage_min, random_env.wage_max, firm.n_wage_levels + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(wage_bin_midpoints, total_offers, color='blue', alpha=0.7, label='Total Firm Offers')
ax.set_title('Firm Wage Offer Frequency (Base Case)')
ax.set_xlabel('Wage Offered')
ax.set_ylabel('Frequency')
ax.legend()
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
ax.set_title('Worker Acceptance Rates by Wage Offer')
ax.set_xlabel('Wage Offered')
ax.set_ylabel('Acceptance Rate')
ax.legend()
plt.tight_layout()
plt.show()


# %%
fig, axes = plt.subplots(figsize=(6, 4))

# Age-Based Model: Firm's Wage Offer Heatmap
im1 = axes.imshow(age_based_results[0].T, origin='lower', aspect='auto', cmap='Blues',
                      extent=[age_based_env.age_start, age_based_env.age_retire, age_based_env.wage_min, age_based_env.wage_max])
axes.set_title('Age-Based: Firm Wage Offer Frequency')
axes.set_xlabel('Worker Age')
axes.set_ylabel('Wage Offered')
fig.colorbar(im1, ax=axes, label='Offer Frequency')

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(figsize=(6, 4))

# Age-Based Model: Worker Acceptance Heatmap
im2 = axes.imshow(age_based_results[1].T, origin='lower', aspect='auto', cmap='coolwarm',
                      extent=[age_based_env.age_start, age_based_env.age_retire, age_based_env.wage_min, age_based_env.wage_max], vmin=0, vmax=1)
axes.set_title('Age-Based: Worker Acceptance Rates')
axes.set_xlabel('Worker Age')
axes.set_ylabel('Wage Offered')
fig.colorbar(im2, ax=axes, label='Acceptance Rate')

plt.tight_layout()
plt.show()


