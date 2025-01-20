# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# defining the environment
class McCallModelEnv:
    def __init__(self, wage_min=0, wage_max=100, value_match=200, age_start=20, age_retire=60,
                  unemployment_penalty=4.0, gamma=1., p_exit=0.01, age_based_mode=False, 
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
        if self.worker_only_mode or self.firm_only_mode or self.random_mode:
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
            self.current_age = np.random.randint(self.age_start, self.age_retire)
            self.reservation_wage = self.age_based_reservation_wage()
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
    
    # add code in simulations and reset/step jugak
    def calculate_reservation_wage(self, gamma, wage_min, wage_max, unemployment_penalty, num_wages=100):
        # calc the dynamic reservation wage based on the McCall model
        wage_range = np.linspace(wage_min, wage_max, num_wages)
        value_function = np.zeros(num_wages)  # Value function initialization
        tolerance = 1e-3
        max_iterations = 1000

        for _ in range(max_iterations):
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
        
    def age_based_reservation_wage(self):
        remaining_work_years = self.age_retire - self.current_age
        max_work_years = self.age_retire - self.age_start
        age_discount_factor = remaining_work_years / max_work_years

        # Base reservation wage from McCall model
        base_reservation_wage = self.calculate_reservation_wage(
            gamma=self.gamma,
            wage_min=self.wage_min,
            wage_max=self.wage_max,
            unemployment_penalty=self.unemployment_penalty,
            num_wages=100
        )
        return base_reservation_wage * age_discount_factor


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
def simulate(env, n_episodes, agent, evaluate_after=20, timesteps_per_sim=100):
    if env.worker_only_mode:
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
                    print("Worker accepted offer of ", wage_offer)
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
        total_firm_offers = np.zeros(firm.n_wage_levels)  # Track total offers by firm
        accepted_firm_offers = np.zeros_like(total_firm_offers)  # Track accepted offers

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
def evaluate(env, agent, timesteps = 100):
    # Reset environment and agent state
    env.reset()
    
    simulations = 50

    acceptance_log = []  # Track all offers and decisions
    for j in range(simulations):
        wage_offers = np.linspace(env.wage_min, env.wage_max, 100)
        np.random.shuffle(wage_offers)
        for i in range(timesteps):
            wage_offer = wage_offers[i]
            state = agent.get_state(wage_offer, env)

            # Use learned policy (no exploration)
            action = 1 if wage_offer >= env.reservation_wage else 0

            # Record decision
            acceptance_log.append((wage_offer, action))

            if action == 1:
                print(f"Accepted Wage Offer: {wage_offer} at Timestep {i}")
                break  # Stop once the agent accepts a wage (as per theory)

    return acceptance_log

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
n_episodes = 200

# -- simulations --

# worker only
wage_bins, eval_acceptance_rates, worker_q_table = simulate(worker_only_env, n_episodes, agent=worker)
eval_results = evaluate(worker_only_env, worker)
eval_wage_offers = [offer for offer, action in eval_results]
eval_acceptances = [action for offer, action in eval_results]

# firm only
total_offers, accepted_offers, acceptance_rates = simulate(firm_only_env, n_episodes, firm)

# mixed firm + worker
firm_offers, worker_accepts, acceptance_rates, firm_profits = simulate(random_env, n_episodes, agent=firm)

age_based_results = simulate(age_based_env, n_episodes, agent=firm)

# %%
# Define finer bins with better alignment
num_bins = 50
wage_bins = np.linspace(worker_only_env.wage_min, worker_only_env.wage_max + 1e-5, num_bins + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

# Process acceptance data
acceptance_counts = np.zeros(num_bins)
offer_counts = np.zeros_like(acceptance_counts)

for offer, accept in zip(eval_wage_offers, eval_acceptances):
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
# firm only env
wage_bins = np.linspace(random_env.wage_min, random_env.wage_max, firm.n_wage_levels + 1)
wage_bin_midpoints = (wage_bins[:-1] + wage_bins[1:]) / 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(wage_bin_midpoints, total_offers, color='blue', alpha=0.7, label='Total Firm Offers')
ax.set_title('Firm Wage Offer Frequency')
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
ax.set_title('Worker Acceptance Rates')
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


