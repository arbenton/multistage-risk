import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from multiprocessing import Pool

class InventoryControlProblem(object):

    def __init__(self, demand_dist, trunc, unit_cost, holding, penalty, setup, revenue, salvage):
        self.lower, self.upper = trunc
        self.demand_dist = demand_dist
        self.c = unit_cost
        self.c_H = holding
        self.c_P = penalty
        self.r = revenue
        self.s = salvage
        self.K = setup

    @property
    def states(self):
        return np.arange(self.lower, self.upper)

    def actions(self, x):
        return [y for y in self.states if y >= x]

    def cost(self, x, y):
        """ deterministic costs from having inventory level x and setting stocking level y. """
        if y not in self.actions(x):
            raise ValueError(f"Stocking level y = {y} is infeasible in state x = {x}.")
        cost = self.c*(y - x) + self.K*int(x < y)
        return cost

    def transition(self, x, y):
        if y not in self.states:
            raise ValueError(f"Stocking level y = {y} greater than UPPER_STATE = {self.upper}.")
        d = np.arange(0, self.upper + 1)
        p = self.demand_dist.pmf(d)
        xp = np.maximum(y - d - self.lower, -self.lower)
        c = self.c_H*np.maximum(y - d, 0) + self.c_P*np.maximum(d - y, 0) - self.r*np.minimum(d, y)
        return c, p, xp

    def terminal_cost(self, x):
        return self.s*x

class DynamicProgrammingSolver(object):
    """ Base class with generalized value iteration """

    def compute_value(self, problem, next_value, x, discount):
        action_value = np.zeros(len(problem.actions(x)))
        best_u_value = np.inf
        for u in problem.actions(x):
            deterministic_cost = problem.cost(x, u)
            random_cost, prob, next_x = problem.transition(x, u)
            risk = self.expect((discount*next_value[next_x] + random_cost), prob)
            action_value = deterministic_cost + risk
            if action_value <= best_u_value:
                best_u = u
                best_u_value = action_value
        return best_u_value, best_u

    def value_iteration(self, problem, nb_period, discount=0.95):
        """ Value Iteration
        params: 
            @problem : object with compliant with Problem API
            @nb_period : number of periods 
            @discount : discount rate 
        returns:
            @value : value function tabulated over state space and time
            @policy : policy function tabulated over states and time
        """
        nb_state = len(problem.states)
        value = np.zeros((nb_period + 1, nb_state))
        policy = np.zeros((nb_period, nb_state))
        value[-1, :] = problem.terminal_cost(problem.states)
        for t in tqdm(reversed(range(0, nb_period)), total=nb_period, desc="Solving Each Stage Subproblem"): 
            for x_idx, x in tqdm(enumerate(problem.states), total=nb_state, leave=False, desc="Iterating over State Space"): 
                value[t, x_idx], policy[t, x_idx] = self.compute_value(problem, 
                                                        value[t + 1], 
                                                        x, 
                                                        discount=discount)
        return value, policy

    def expect(self, v_, p_):
        raise NotImplementedError("expect method should be implemented by child class.")

class ExpectationSolver(DynamicProgrammingSolver):
    """ Expectation Minimization Solver """

    def expect(self, v_, p_):
        return v_.dot(p_)

class DualMeanSemideviationSolver(DynamicProgrammingSolver):
    
    def __init__(self, nb_measure, q=2, c=1):
        self.h = cp.Variable(nb_measure)
        self.p = cp.Parameter(nb_measure, nonneg=True)
        self.v = cp.Parameter(nb_measure)
        self.g = 1 + self.h - cp.matmul(self.h, self.p)
        objective = cp.Maximize(cp.matmul(self.v, cp.multiply(self.g, self.p)))
        constraints = [
            self.g >= 0,
            cp.matmul(cp.power(self.h, q), self.p) <= c**q
        ]
        self.problem = cp.Problem(objective, constraints)

    def expect(self, v_, p_):
        self.v.value = v_
        self.p.value = p_
        risk = self.problem.solve(warm_start=True)
        return risk

class MeanSemideviationSolver(DynamicProgrammingSolver):
    
    def __init__(self, nb_measure, q=2, c=1):
        self.nb_measure = nb_measure 
        self.q = q
        self.c = c

    def expect(self, v, p):
        mean = v.dot(p)
        semi = np.power(np.power(v - mean, self.q).dot(p), 1/self.q)
        risk = mean + self.c*semi
        return risk

class DualMeanUpperSemideviationSolver(DynamicProgrammingSolver):
    
    def __init__(self, nb_measure, q=2, c=1):
        self.h = cp.Variable(nb_measure, nonneg=True)
        self.p = cp.Parameter(nb_measure, nonneg=True)
        self.v = cp.Parameter(nb_measure)
        self.g = cp.multiply(self.p, 1 + self.h - cp.matmul(self.h, self.p))
        objective = cp.Maximize(cp.matmul(self.v, self.g))
        constraints = [
            self.g >= 0,
            cp.matmul(cp.power(self.h, q), self.p) <= c**q
        ]
        self.problem = cp.Problem(objective, constraints)

    def expect(self, v_, p_):
        self.v.value = v_
        self.p.value = p_
        risk = self.problem.solve(warm_start=True, solver=cp.ECOS)
        if abs(np.sum(self.g.value) - 1) > 1e-3:
            raise UserWarning("Subgradient is not a probability distribution")
        return risk

class MeanUpperSemideviationSolver(DynamicProgrammingSolver):
    
    def __init__(self, nb_measure, q=2, c=1):
        self.nb_measure = nb_measure 
        self.q = q
        self.c = c

    def expect(self, v, p):
        mean = v.dot(p)
        semi = np.power(np.power(np.maximum(v - mean, 0), self.q).dot(p), 1/self.q)
        risk = mean + self.c*semi
        return risk

class DualCVaRSolver(DynamicProgrammingSolver):

    def __init__(self, nb_measure, alpha=0.95):
        self.h = cp.Variable(nb_measure, nonneg=True)
        self.p = cp.Parameter(nb_measure, nonneg=True)
        self.v = cp.Parameter(nb_measure)
        objective = cp.Maximize(cp.matmul(self.v, cp.multiply(self.h, self.p)))
        constraints = [
            self.h <= 1/(1 - alpha),
            cp.matmul(self.h, self.p) == 1
        ]
        self.problem = cp.Problem(objective, constraints)

    def expect(self, v_, p_):
        self.v.value = v_
        self.p.value = p_
        risk = self.problem.solve(solver=cp.SCS, warm_start=True)
        return risk

class CVaRSolver(DynamicProgrammingSolver):

    def __init__(self, nb_measure, alpha=0.95):
        self.nb_measure = nb_measure 
        self.alpha = alpha

    def expect(self, v, p):
        q = np.percentile(v, 100*self.alpha)
        risk = np.mean(v[np.where(v >= q)])
        return risk


class EVaRSolver(DynamicProgrammingSolver):

    def __init__(self, nb_measure, alpha=0.95):
        self.alpha = alpha
        self.z = cp.Variable(nb_measure, nonneg=True)
        self.p = cp.Parameter(nb_measure, nonneg=True)
        self.v = cp.Parameter(nb_measure)
        dkl = cp.matmul(self.p, -cp.entr(self.z))
        objective = cp.Maximize(cp.matmul(self.p, cp.multiply(self.v, self.z)))
        constraints = [
            dkl <= np.log(1/(1 - self.alpha)),
            cp.matmul(self.p, self.z) == 1
        ]
        self.problem = cp.Problem(objective, constraints)

    def expect(self, v_, p_):
        self.v.value = v_
        self.p.value = p_
        risk = self.problem.solve(solver=cp.SCS, warm_start=True)
        if risk is None:
            risk = np.inf
        return risk

if __name__ == '__main__':
    nb_period = 3
    max_demand = 20
    upper_state = int(max_demand)
    lower_state = 0
    lambd = 1
    demand_dist = stats.randint(0, 4)
    problem = InventoryControlProblem(demand_dist, 
                                      (lower_state, upper_state), 
                                      unit_cost=1.0, 
                                      holding=0.5,
                                      penalty=1.0,
                                      setup=0.0,
                                      revenue=2.0,
                                      salvage=-1.0)#1.0)
    #solver = EVaRSolver(len(problem.states), alpha=0.90)
    #solver = CVaRSolver(len(problem.states), alpha=0.90)
    solver = ExpectationSolver()
    value, policy = solver.value_iteration(problem, nb_period, discount=1.0)
    print(policy[:, 0])
    print(value[:, 0])

    plt.subplot(2, 1, 1)
    for t in range(nb_period):
        x = np.arange(lower_state + 1, upper_state + 1)
        plt.plot(x, policy[t, :], label=f"t = {t}")
    plt.ylabel("Order Up To (y)")
    plt.legend()
    plt.subplot(2, 1, 2)
    for t in range(nb_period):
        x = np.arange(lower_state + 1, upper_state + 1)
        plt.plot(x, value[t, :], label=f"t = {t}")
    plt.legend()
    plt.xlabel("Starting Inventory (x)")
    plt.savefig("base-stock-policy.png")
