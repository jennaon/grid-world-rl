from gridworld import GridWorldMDP
from qlearn import QLearner

import numpy as np
import matplotlib.pyplot as plt
import pdb
import random


def plot_convergence(utility_grids, policy_grids):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1,2))
    # pdb.set_trace()
    ax1.plot(utility_ssd, 'b.-')
    ax1.set_ylabel('Change in Utility', color='b')

    policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1,2))
    ax2.plot(policy_changes, 'r.-')
    ax2.set_ylabel('Change in Best Policy', color='r')


if __name__ == '__main__':
    np.random.seed(31)
    random.seed(29)
    shape = (4,20,7)
    # shape = (2,5,5)
    goal_x = [6,4,2]
    goal_y = [3,2,12]
    goal_z = [2,1,3]
    # goal_x = [0]
    # goal_y = [1]
    # goal_z= [0]

    # num_obstacles = 5
    num_obstacles = 200
    # trap = (1, 3,0)

    #no glory for now
    # obstacle_x = 0
    # obstacle_y = 0
    # obstacle_z = 0
    obstacles=random.sample(range(shape[0]*shape[1]*shape[2]),num_obstacles)
    obstacle_z, obstacle_y, obstacle_x = np.unravel_index(obstacles, (shape[0], shape[1],shape[2]))

    start = (3, 16, 5) #fixed for now
    default_reward = -.1
    goal_reward = 100
    # trap_reward = -10 #add traps later maybe

    reward_grid = np.zeros(shape) + default_reward
    reward_grid[goal_z, goal_y, goal_x] = goal_reward
    # reward_grid[trap] = trap_reward
    reward_grid[obstacle_z, obstacle_y, obstacle_x] = -1
    # pdb.set_trace()

    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    terminal_mask[goal_z, goal_y, goal_x] = True
    # terminal_mask[trap] = True

    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
    obstacle_mask[obstacle_z, obstacle_y, obstacle_x] = True

    # pdb.set_trace()
    gw = GridWorldMDP(start = start,
                      reward_grid=reward_grid,
                      obstacle_mask=obstacle_mask,
                      terminal_mask=terminal_mask,
                      action_probabilities=[
                      # up, right, down, left, float, sink, stay
                          (0,.8),
                          (1,.8),
                          (2,.6),
                          (3,.6),
                          (4,.9),
                          (5,.9),
                          (6,.2)
                      ],
                      no_action_probability=0.0)

    # -----For Value Iteration -----#
    # mdp_solvers = {'Value Iteration': gw.run_value_iterations}
    # discount_ =1
    # for solver_name, solver_fn in mdp_solvers.items():
    #     print('Final result of {}:'.format(solver_name))
    #     np.set_printoptions(precision=1)
    #     policy_grids, utility_grids = solver_fn(iterations=100 , discount=discount_)
    #     # pdb.set_trace()
    #     # print(policy_grids[:, :, :, -1])
    #     # print(utility_grids[:, :, :, -1])
    #
    #     gw.plot_policy(utility_grids[:, :, :, -1], discount_)
    #     plot_convergence(utility_grids, policy_grids)
    #     # plt.show()
    #
    # best_reward, good_coords = gw.evaluate(start, policy_grids[:,:,:,-1], eval_iters=20)
    # gw.plot_path(good_coords, best_reward,utility_grids[:,:,:,-1])
    # print('best reward with value iteration: %.2f'%best_reward) #92.70
    # pdb.set_trace()
    # -----End Value Iteration -----#
    '''
    parm
    '''
    # pdb.set_trace()
    gamma = 1
    # -----For Q learning -----#
    #
    np.set_printoptions(precision=3)
    ql = QLearner(state_shape=shape,
                  num_actions=7,
                  learning_rate=0.1,
                  discount_rate=gamma,
                  random_action_prob=0.2,
                  random_action_decay_rate=0.99,
                  dyna_iterations=0)

    start_state = gw.grid_coordinates_to_indices(start)

    iterations = 1000
    flat_policies, flat_utilities, end_iter = ql.learn(gw,start_state,
                                             gw.generate_experience,
                                             gw._get_reward,
                                             iterations=iterations)

    ql_utility_grids = flat_utilities
    ql_policy_grids = flat_policies
    # # new_shape = (gw.shape[0], gw.shape[1], gw.shape[2], end_iter)
    # # ql_utility_grids = flat_utilities.reshape(new_shape)
    # # ql_policy_grids = flat_policies.reshape(new_shape)
    # # print('Final result of QLearning:')
    # # print(ql_policy_grids[:, :, :, -1])
    # # print(ql_utility_grids[:, :, :, -1])


    print('evaluating.....')
    best_reward, good_coords= gw.evaluate(start, ql_policy_grids[:,:,:,-1] )


    # gw.plot_path(path, total_reward, ql_utility_grids)
    print('best reward : %.2f'%best_reward) #95.20

    gw.plot_policy(ql_utility_grids[:, :, :, -1], gamma, ql_policy_grids[:, :, :, -1])
    # plt.suptitle('U_min= %.2f, U_max = %.2f, max iter %d' %(umin, umax, end_iter))
    plot_convergence(ql_utility_grids, ql_policy_grids)
    plt.show()
    pdb.set_trace()
    # -----END Q learning -----#
