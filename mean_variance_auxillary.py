import numpy as np
import matplotlib.pyplot as plt
import copy
#Assume that the terminal profit is always 0.
#For discrete distribution
possible_value = [0, 1, 2, 3]
prob = [0.25, 0.25, 0.25, 0.25]
dist = dict(zip(possible_value, prob))

# Basic Settings
T = 3
c_H = 0.2   #Holding cost
c_P = 2.0   #Panalty for shortage
unit_cost = 1.0
price =  5.0 #selling price
lambd =  0.1
beta = 0.01
c_SALVAGE = 0 #1.5



def possible_combo_generator(num, dividend): #can be used to generate all combinations of demand, all combinations of base stock.
    l = []
    while num > 0:
        num, remainder = num // dividend, num % dividend
        l.append(remainder)
    states = l[::-1]
    return [0] * (T - len(states)) + states

#Generate inventory and order up_to list
def y_level(x_level, policy_level):
    return max(x_level, policy_level)

def x_level_next(y_level, Demand):
    return (y_level - Demand if y_level - Demand >= 0 else 0)


def x_y_combo_list(x_1, policy_list, demand_list): #return the corresbonding x_list and y_list
    x_y_list = [x_1] + [0] * 2 * len(policy_list)
    for i in range(1, len(x_y_list)):
        if i % 2 == 0:
            x_y_list[i] = x_level_next(x_y_list[i-1], demand_list[int((i-2)/2)])
        else:
            x_y_list[i] = y_level(x_y_list[i-1], policy_list[int((i-1)/2)])
    return [x_y_list[i] for i in range(len(x_y_list)) if i % 2 == 0], [x_y_list[i] for i in range(len(x_y_list)) if i % 2 != 0]

"""
def x_level_list(x_1, policy_list, demand_list):
    if len(policy_list) == 0:
        return [x_1]
    else:
        for t in range(len(policy_list)):
            x_next = x_level_next(y_level(x_list[t], policy_list[t]), demand_list[t])
            x_list.append(x_next)
        return x_level_list(x_1, policy_list, demand_list)

def y_level_list(x_1, policy_list, demand_list):
    y_list = []
    for t in range(len(policy_list)):
        y = y_level(x_level_list[t], demand_list[t])
        y_list.append(y)
    return y_list
"""

def profit_single_period(x_level, y_level, Demand): 
    if y_level >= Demand:
        profit = unit_cost * (x_level - y_level) - c_H * (y_level - Demand) + price * Demand 
    elif y_level < Demand:
        profit = unit_cost * (x_level - y_level) - c_P * (Demand - y_level) + price * y_level 
    return profit

def cost_single_period(x_level, y_level, Demand):
    if y_level >= Demand:
        cost = unit_cost * (y_level - x_level) + c_H * (y_level - Demand) 
    elif y_level < Demand:
        cost = unit_cost * (y_level - x_level) + c_P * (Demand - y_level) 
    return cost


def sum_profit_prob_1(x_list, y_list, demand_list): 
    """return the sum of profit and the pro of getting that profit with determined x_1 and policy_list. Ignore salvage."""
    profit = c_SALVAGE * x_list[-1]
    prob = 1
    for t in range(len(y_list)):
        profit = profit + profit_single_period(x_list[t], y_list[t], demand_list[t])
        prob = prob * dist[demand_list[t]]
    return profit, prob

def sum_profit_prob_2(x_list, y_list, demand_list): #negative cost as the objective
    """return the sum of profit and the pro of getting that profit with determined x_1 and policy_list. Ignore salvage."""
    profit = c_SALVAGE * x_list[-1]
    prob = 1
    for t in range(len(y_list)):
        profit = profit + profit_single_period(x_list[t], y_list[t], demand_list[t])
        prob = prob * dist[demand_list[t]]
    return profit, prob

"""
state_action_combo = []
for x_1 in range(11):
    for number in range(11 ** T):
        action = state_action_generator(number, 11)
        if x_1 <= action[0]:
            state_action_combo.append([x_1, action])
"""

#Need to be revised if T and dist changed
all_path = {} 
for x_1 in range(11):
    x_policy_dict = {}
    for policy_gene in range(11 ** T):   #generate policy_list
        policy_list = possible_combo_generator(policy_gene, 11)
        profit_prob_combo = []
        for demand_gene in range(len(dist) ** T):
            demand_list = possible_combo_generator(demand_gene, len(dist))
            x_list, y_list = x_y_combo_list(x_1, policy_list, demand_list)
            profit, prob = sum_profit_prob(x_list, y_list, demand_list)
            profit_prob_combo.append([profit, prob])
        x_policy_dict[tuple(policy_list)] = profit_prob_combo
    all_path[x_1] = x_policy_dict


all_mean_variance = {} 
for x_1, x_policy_dict in all_path.items():
    policy_mean_dict = {}
    for policy, policy_dict in x_policy_dict.items():
        mean = 0
        mean_square = 0
        for profit_prob in policy_dict:
            mean += profit_prob[0] * profit_prob[1]
            mean_square += (profit_prob[0]) ** 2 * profit_prob[1]
        variance = mean_square - mean ** 2
        mean_risk = mean - lambd * variance
        policy_mean_dict[policy] = [mean, variance, mean_risk]
    all_mean_variance[x_1] = policy_mean_dict


#EXPECTATION
all_mean = copy.deepcopy(all_mean_variance)
for x_1, x_policy_dict in all_mean.items():
    for key, value in x_policy_dict.items(): 
        x_policy_dict[key] = value[0]

best_strategy_expection = {}
for x_1, x_policy_dict in all_mean.items(): 
    best_strategy = {}
    for key, value in x_policy_dict.items(): #can find all the keys corresponds to values 
        if (value == max(x_policy_dict.values())):
            best_strategy[key] = value
    best_strategy_expection[x_1] = best_strategy

#MEAN_VARIANCE
all_mv = copy.deepcopy(all_mean_variance)
for x_1, x_policy_dict in all_mv.items():
    for key, value in x_policy_dict.items(): 
        x_policy_dict[key] = value[2]

best_strategy_mv = {}
for x_1, x_policy_dict in all_mv.items(): 
    best_strategy = {}
    for key, value in x_policy_dict.items(): #can find all the keys corresponds to values 
        if (value == max(x_policy_dict.values())):
            best_strategy[key] = value
    best_strategy_mv[x_1] = best_strategy


#Auxillary Problem
auxillary_variance = {} 
for x_1, x_policy_dict in all_path.items():
    policy_mean_dict = {}
    for policy, policy_dict in x_policy_dict.items():
        mean = 0
        mean_square = 0
        for profit_prob in policy_dict:
            mean += profit_prob[0] * profit_prob[1]
            mean_square += (profit_prob[0]) ** 2 * profit_prob[1]
        mean_risk = mean - beta * mean_square
        policy_mean_dict[policy] = mean_risk
    auxillary_variance[x_1] = policy_mean_dict


best_strategy_auxillary = {}
for x_1, x_policy_dict in auxillary_variance.items(): 
    best_strategy = {}
    for key, value in x_policy_dict.items(): #can find all the keys corresponds to values 
        if (value == max(x_policy_dict.values())):
            best_strategy[key] = value
    best_strategy_auxillary[x_1] = best_strategy






