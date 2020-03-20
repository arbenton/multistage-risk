import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from multiprocessing import Pool
from random import randint

import gym

class InventoryControlEnvironment(gym.Env):

    def __init__(self, demand_dist, trunc, unit_cost=1, holding=0.5, penalty=5.0, setup=0, horizon=100, discount=1):
        self.lower, self.upper = trunc
        self.demand_dist = demand_dist
        self.c = unit_cost
        self.c_H = holding
        self.c_P = penalty
        self.K = setup
        self.T = horizon
        self.discount = discount
        self.state_space = gym.spaces.Tuple((gym.spaces.Discrete(self.T), gym.spaces.Box(self.lower, self.upper, (1, ))))
        self.action_space = gym.spaces.Box(self.lower, self.upper, (1, ))

    @property
    def states(self):
        return np.arange(self.lower, self.upper)

    def actions(self, x):
        return [y for y in self.states if y >= x]

    def state_action_cost(self, x, y):
        """ deterministic costs from having inventory level x and setting stocking level y. """
        if y not in self.actions(x):
            raise ValueError(f"Stocking level y = {y} is infeasible in state x = {x}.")
        cost = self.c*(x - y) + self.K*int(x < y)
        return cost

    def terminal_cost(self, x):
        return self.c_H*np.maximum(x, 0) + self.c_P*np.maximum(x, 0)

    def reset(self):
        self.x = randint(self.lower, self.upper - 1)
        self.t = 0
        return (self.t, self.x)

    def step(self, y):
        cost = self.state_action_cost(self.x, y)
        self.x = y
        d = self.demand_dist.rvs()
        cost += self.c_H*np.maximum(y - d, 0) + self.c_P*np.maximum(d - y, 0)
        self.x = max(self.lower, self.x - d)
        self.t += 1
        if self.t == self.T:
            cost += self.discount*self.terminal_cost(self.x)
            done = True
        else:
            done = False
        return (self.t, self.x), -cost, done, {}

if __name__ == '__main__':
    import radp
    from tqdm import tqdm
    from collections import defaultdict

    nb_period = 10
    discount = 1.0
    max_demand = 50
    upper_state = int(max_demand)
    lower_state = int(-max_demand)
    lambd = 1
    demand_dist = stats.poisson(lambd)
    problem = radp.InventoryControlProblem(demand_dist, 
                                           (lower_state, upper_state), 
                                           unit_cost=1, 
                                           holding=0.1, 
                                           penalty=1.5,
                                           setup=0.0)

    solver = radp.ExpectationSolver()
    value, policy = solver.value_iteration(problem, nb_period, discount=discount)
        
    nb_samples = 50000

    env = InventoryControlEnvironment(demand_dist, 
                                      (lower_state, upper_state),
                                      unit_cost=1,
                                      holding=0.1,
                                      penalty=1.5,
                                      setup=0.0,
                                      discount=discount,
                                      horizon=nb_period)

    value_samples = defaultdict(list)
    for _ in tqdm(range(nb_samples), desc="Simulating policy"):
        n = 0
        t, x = env.reset()
        initial_state = x
        discounted_rewards = 0
        done = False
        while not done:
            y = policy[t, int(x - lower_state)]
            (t, x), reward, done, _ = env.step(y)
            discounted_rewards += discount**n*reward
            n += 1
        value_samples[initial_state].append(discounted_rewards)

    simulated_value = np.zeros(len(value_samples))
    simulated_value_lower1 = np.zeros(len(value_samples))
    simulated_value_upper1 = np.zeros(len(value_samples))
    
    simulated_value_lower2 = np.zeros(len(value_samples))
    simulated_value_upper2 = np.zeros(len(value_samples))
    for x in sorted(value_samples):
        rewards = value_samples[x]
        #print(x, -np.mean(rewards), value[0, int(x - lower_state)], len(rewards))
        simulated_value[int(x - lower_state)] = -np.mean(rewards)
        simulated_value_lower1[int(x - lower_state)] = -np.percentile(rewards, 97.5)
        simulated_value_upper1[int(x - lower_state)] = -np.percentile(rewards, 2.5)
        simulated_value_lower2[int(x - lower_state)] = -np.percentile(rewards, 99.5)
        simulated_value_upper2[int(x - lower_state)] = -np.percentile(rewards, 0.5)

    print(simulated_value)
    x = np.arange(lower_state, upper_state)
    fig, ax = plt.subplots()
    ax.fill_between(x, simulated_value_lower2, simulated_value_upper2, color="black", alpha=0.1, label="Simulated 99% CI")
    ax.fill_between(x, simulated_value_lower1, simulated_value_upper1, color="black", alpha=0.1, label="Simulated 95% CI")
    ax.plot(x, value[0, :], "g-", label="True Value")
    ax.plot(x, simulated_value, "r--", label="Simulated Value")
    ax.set_xlabel(r"Inventory Level")
    ax.set_ylabel(r"Discounted Costs")
    ax.legend()
    fig.savefig("initial_value.png", dpi=300)

    x = upper_state//2
    costs = [-v for v in value_samples[x]]
    
    params = stats.norm.fit(costs)
    vv = np.linspace(np.min(costs), np.max(costs), 1000)
    pdf = stats.norm.pdf(vv, *params)
    fig, ax = plt.subplots()
    ax.hist(costs, 50, density=True, color="white", alpha=0.9, ec="black")
    ax.plot(vv, pdf, "r--")
    ax.set_xlabel(r"Discounted Costs")
    #ax.axvline(value[0, x + lower_state])
    fig.savefig("histogram.png", dpi=300)