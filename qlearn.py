import numpy as np
import random as rand
import pdb

class QLearner:
    '''A generic implementation of Q-Learning and Dyna-Q'''

    def __init__(self, *,
                 state_shape,
                 num_actions,
                 learning_rate,
                 discount_rate=1.0,
                 random_action_prob=0.5,
                 random_action_decay_rate=0.99,
                 dyna_iterations=0):
        self.shape = state_shape
        self._num_states = self.shape[0] * self.shape[1] * self.shape[2]
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._random_action_prob = random_action_prob
        self._random_action_decay_rate = random_action_decay_rate
        self._dyna_iterations = dyna_iterations
        self.lambd = 0.4
        self.bottom = 0.001
        self.eps = .001


        self._experiences = []

        # Initialize Q to small random values.
        self._Q = np.zeros((self._num_states, num_actions), dtype=np.float)
        # self._Q += np.random.normal(0, 0.3, self._Q.shape)


    def initialize(self, state):
        '''Set the initial state and return the learner's first action'''
        self._stored_action = self._decide_next_action(state)
        self._stored_state = state
        self._M = np.zeros_like(self._Q) #for sarsa-lambda
        return self._stored_action

    def learn(self, gw,initial_state, experience_func, reward_func, iterations=100):
        '''Iteratively experience new states and rewards'''
        Q_prev = np.copy(self._Q)-float('inf')
        H,M,N = self.shape
        all_policies = np.zeros((H,M,N, iterations))
        all_utilities = np.zeros_like(all_policies)
        for i in range(iterations):
            done = False
            self.initialize(initial_state)
            move_count = 0
            while True:
                move_count +=1
                # if i == 47:
                #     print(next_state, self._stored_action)
                next_state, reward, done = experience_func(self._stored_state,
                                                      self._stored_action)
                # if i == 47 and next_state == 517:
                #     pdb.set_trace()
                #adjust your Q values based on your new experience
                if next_state == None:
                    self._Q[self._stored_state,:] += reward
                    print('cul de sac :( )')
                    break
                elif move_count > 500:#you're probably stuck somewhere, which is equivalent of cul de sac
                    self._Q[self._stored_state, self._stored_action] += self._learning_rate * -100
                    self._Q[next_state, :] += self._learning_rate * -100
                    print('!!stuck!')
                    break
                else:
                    self.experience(next_state, reward)
                    if done:
                        # self._M[next_state,self._stored_action] += 1
                        # delta = reward_func(next_state, self._stored_action)#\
                        #             #-self_Q[self._stored_state, self._stored_action]
                        # self._Q += self._learning_rate * delta * self._M
                        # self._M *= self._discount_rate*self.lambd
                        #sarsa
                        r = reward_func(next_state, self._stored_action)#-self_Q[self._stored_state, self._stored_action]
                        self._Q[next_state,self._stored_action] += self._learning_rate * r
                        break

            policy, utility = self.get_policy_and_utility()
            all_policies[:, :, :, i] = policy.reshape((self.shape))
            all_utilities[:, :, :, i] = utility.reshape((self.shape))

            if i > 20 and np.linalg.norm( (self._Q-Q_prev), ord=np.inf ) <self.eps :
                print('converged at iteration %d'%i)
                policy, utility = self.get_policy_and_utility()
                all_policies[:, :, :, i] = policy.reshape((self.shape))
                all_utilities[:, :, :, i] = utility.reshape((self.shape))
                return all_policies[:,:,:,:i], all_policies[:,:,:,:i], i
            # if np.mod(i,10) == 0:
            print('iter %d, norm %.2f'% (i,np.linalg.norm( (self._Q-Q_prev), ord=np.inf )) )
            Q_prev = np.copy(self._Q)
            # pdb.set_trace()
        print('loop broke by safety lock')
        return all_policies, all_utilities, iterations



    def experience(self, next_state, reward):
        '''The learner experiences state and receives a reward'''

        # if self._dyna_iterations > 0:
        #     print('random code running')
        #     self._experiences.append(
        #         (self._stored_state, self._stored_action, state, reward)
        #     )
        #     exp_idx = np.random.choice(len(self._experiences),
        #                                self._dyna_iterations)
        #     for i in exp_idx:
        #         self._update_Q(*self._experiences[i])

        # determine an action and update the current state
        next_action = self._decide_next_action(next_state)
        self._update_Q(self._stored_state, self._stored_action, reward, next_state, next_action)

        self._stored_state = next_state
        self._stored_action = next_action
        self._random_action_prob *= self._random_action_decay_rate

        # return self._stored_action

    def get_policy_and_utility(self):
        policy = np.argmax(self._Q, axis=1)
        utility = np.max(self._Q, axis=1)
        return policy, utility

    def _update_Q(self, s, a, r, sp, ap):
        # # sarsa lambda
        # self._M[s,a] += 1
        # delta = r + self._discount_rate * self._Q[sp, ap]-self._Q[s,a]
        # self._Q += self._learning_rate * delta * self._M
        # self._M *= self._discount_rate*self.lambd
        # self._M[self._M<self.bottom] = 0
        # update terminal condition as well
        #
        #sarsa
        delta = r + self._discount_rate * self._Q[sp, ap]-self._Q[s,a]
        self._Q[s,a] += self._learning_rate * delta
        # # pdb.set_trace()



        # best_reward = self._Q[s_prime, self._find_best_action(s_prime)]
        # self._Q[s, a] *= (1 - self._learning_rate)
        # self._Q[s, a] += (self._learning_rate
        #                   * (r + self._discount_rate * best_reward))

    def _decide_next_action(self, state): #epsilon_greedy
        if rand.random() <= self._random_action_prob:
            return rand.randint(0, self._num_actions - 1)
        else:
            return self._find_best_action(state)

    def _find_best_action(self, state):
        # pdb.set_trace()
        return int(np.argmax(self._Q[state, :]))
