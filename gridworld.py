import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import ast
EPSILON = 0.1

class GridWorldMDP:
    # np.seterr(divide='ignore', invalid='ignore')

    # up, right, down, left, float, sink, stay
    _direction_deltas = [
        #z, y, x
        (0, -1, 0), #0 up
        (0, 0, 1),  #1 right
        (0, 1, 0),  #2 down
        (0, 0, -1), #3 left
        (1, 0, 0),  #4 float
        (-1, 0, 0), #5 sink
        (0, 0, 0)   #6 stay
    ]
    _num_actions = len(_direction_deltas)
    '''
    (0, -1, 0), #0 up
    (0, 0, 1),  #1 right
    (0, 1, 0),  #2 down
    (0, 0, -1), #3 left
    '''
    landmark_lookup={
    #(obstacle,edges)->observation index)
    '(None, None)':0,
    '(0, None)':1,
    '(1, None)':2,
    '(2, None)':3,
    '(3, None)':4,
    '(None, 0)':5,
    '(None, 1)':6,
    '(None, 2)':7,
    '(None, 3)':8,
    '(None, (0, 3))':9,
    '(None, (0, 1))':10,
    '(None, (1, 2))':11,
    '(None, (2, 3))':12,
    '(0, 1)':13,
    '(0, 2)':14,
    '(0, 3)':15,
    '(0, (1, 2))':16,
    '(0, (2, 3))':17,
    '(1, 0)':18,
    '(1, 2)':19,
    '(1, 3)':20,
    '(1, (0, 3))':21,
    '(1, (2, 3))':22,
    '(2, 0)':23,
    '(2, 1)':24,
    '(2, 3)':25,
    '(2, (0, 3))':26,
    '(2, (0, 1))':27,
    '(3, 0)':28,
    '(3, 1)':29,
    '(3, 2)':30,
    '(3, (0, 1))':31,
    '(3, (1, 2))':32,
    '((0, 1), None)':33,
    '((0, 1), 2)':34,
    '((0, 1), 3)':35,
    '((0, 1), (2, 3))':36,
    '((0, 2), None)':37,
    '((0, 2), 1)':38,
    '((0, 2), 3)':39,
    '((0, 3), None)':40,
    '((0, 3), 1)':41,
    '((0, 3), 2)':42,
    '((0, 3), (1, 2))':43,
    '((1, 2), None)':44,
    '((1, 2), 0)':45,
    '((1, 2), 3)':46,
    '((1, 2), (0, 3))':47,
    '((1, 3), None)':48,
    '((1, 3), 0)':49,
    '((1, 3), 2)':50,
    '((2, 3), None)':51,
    '((2, 3), 0)':52,
    '((2, 3), 1)':53,
    '((2, 3), (0, 1))':54,
    '((0, 1, 2), None)': 55,
    '((0, 1, 2), 3)': 56,
    '((0, 1, 3), None)': 57,
    '((0, 1, 3), 2)': 58,
    '((1, 2, 3), None)': 59,
    '((1, 2, 3), 0)': 60,
    '((0, 2, 3), None)': 61,
    '((0, 2, 3), 1)': 62,
    '((0, 1, 2, 3), None)': 63
    }
    landmark_idx_lookup = {v: k for k, v in landmark_lookup.items()}

    def __init__(self,
                 start,
                 reward_grid,
                 terminal_mask,
                 obstacle_mask,
                 action_probabilities,
                 no_action_probability):
        self.start = start
        self._reward_grid = reward_grid
        self._terminal_mask = terminal_mask
        self._obstacle_mask = obstacle_mask
        self._T = self._create_transition_matrix(
            action_probabilities,
            no_action_probability,
            obstacle_mask
        )
        self.num_obs = 64
        self.meas_accuracy = 0.6

        set0 = {'(None, None)',
                '(0, None)',
                '(1, None)',
                '(2, None)',
                '(3, None)',
                '((0, 1), None)',
                '((0, 2), None)',
                '((0, 3), None)',
                '((1, 2), None)',
                '((1, 3), None)',
                '((2, 3), None)',
                '((0, 1, 2), None)',
                '((0, 1, 3), None)',
                '((1, 2, 3), None)',
                '((0, 2, 3), None)',
                '((0, 1, 2, 3), None)'}
        set1 = {'(None, 0)', '(1, 0)','(2, 0)','(3, 0)',
                '((1, 2), 0)',
                '((1, 3), 0)',
                '((2, 3), 0)',
                '((1, 2, 3), 0)'}
        set2 = {'(None, 1)',
                '(0, 1)',
                '(2, 1)',
                '(3, 1)',
                '((0, 2), 1)',
                '((0, 3), 1)',
                '((2, 3), 1)',
                '((0, 2, 3), 1)'}

        set3 = {'(None, 2)',
                '(0, 2)',
                '(1, 2)',
                '(3, 2)',
                '((0, 1), 2)',
                '((0, 3), 2)',
                '((1, 3), 2)',
                '((0, 1, 3), 2)'}
        set4 = {'(None, 3)',
                '(0, 3)',
                '(1, 3)',
                '(2, 3)',
                '((0, 1), 3)',
                '((0, 2), 3)',
                '((1, 2), 3)',
                '((0, 1, 2), 3)'}

        set5 = {'(None, (0, 1))',
                '(2, (0, 1))',
                '(3, (0, 1))',
                '((2, 3), (0, 1))'}

        set6 = {'(None, (1, 2))',
                '(0, (1, 2))',
                '(3, (1, 2))',
                '((0, 3), (1, 2))'}

        set7 = {'(None, (2, 3))',
                '(0, (2, 3))',
                '(1, (2, 3))',
                '((0, 1), (2, 3))'}

        set8 = {'(None, (0, 3))',
                '(1, (0, 3))',
                '(2, (0, 3))',
                '((1, 2), (0, 3))'}
        self.sets = [set0,set1, set2, set3, set4,set5, set6, set7, set8]

        self._O = self._create_observation_matrix()
        self.build_neighbor_dict()


    @property
    def shape(self):
        return self._reward_grid.shape

    @property
    def size(self):
        return self._reward_grid.size

    @property
    def reward_grid(self):
        return self._reward_grid

    def run_value_iterations(self, discount=1.0,
                             iterations=10):
        utility_grids, policy_grids= self._init_utility_policy_storage(iterations)
        utility_grid = np.zeros_like(self._reward_grid)
        iter = 0
        prev_utility = np.zeros_like(utility_grid)
        while True:
            # print('-----------------------------------')
            utility_grid = self._value_iteration(utility_grid, discount)
            # print('outside values:')
            # print(utility_grid)
            # pdb.set_trace()
            policy_grids[:, :, :, iter] = self.best_policy(utility_grid, discount)
            utility_grids[:, :, :, iter] = utility_grid


            # print('iter %d'%(iter))
            if np.abs( np.linalg.norm(utility_grid-prev_utility)) < EPSILON:
                return policy_grids[:,:,:,:iter], utility_grids[:,:,:,:iter]

            if iter+1==iterations:
                print('iteration broke by safety loop')
                return policy_grids, utility_grids
            # print(utility_grid)
            # pdb.set_trace()

            prev_utility = np.copy(utility_grid)
            iter +=1

    def run_policy_iterations(self, discount=1.0,
                              iterations=10):
        utility_grids, policy_grids = self._init_utility_policy_storage(iterations)

        policy_grid = np.random.randint(0, self._num_actions,
                                        self.shape)
        utility_grid = self._reward_grid.copy()

        for i in range(iterations):
            print(i)
            policy_grid, utility_grid = self._policy_iteration(
                policy_grid=policy_grid,
                utility_grid=utility_grid
            )
            policy_grids[:, :, i] = policy_grid
            utility_grids[:, :, i] = utility_grid
        return policy_grids, utility_grids

    def evaluate(self,initial_state, policy_grid,eval_iters =100):
        H,M,N = self.shape

        good_coords = []
        best_reward= -float('inf')
        experience_count = 0
        while True:
            print('%d th experience....'%experience_count)
            done = False
            current_state = initial_state
            current_action = policy_grid[current_state]
            count = 0
            total_reward = self._get_reward(current_state, current_action)

            new_coords = []
            move_count = 0
            while not done :
                move_count +=1
                new_coords.append(current_state)

                next_state, reward, done = self.generate_experience_with_grid(current_state, int(current_action))
                if next_state == None:
                    total_reward += reward
                    print('cul de sac!!!')
                    break
                next_action = policy_grid[next_state]
                total_reward += reward

                if done:
                    print('end!')
                    total_reward += self._reward_grid[next_state]
                    new_coords.append(current_state)
                    break
                elif move_count >500:
                    total_reward -=100
                    break
                else:
                    # new_coords.append(current_state[::-1])
                    current_state = next_state
                    current_action = next_action

            if total_reward > best_reward:
                good_coords = np.copy(new_coords)
                best_reward = total_reward
            # pdb.set_trace()
            experience_count +=1
            if experience_count == eval_iters :
                return best_reward, good_coords

    def generate_experience_with_grid(self, current_state, action):
        sz, sy, sx = current_state
        next_state_probs = self._T[sz, sy, sx, action, :, :, :].flatten()
        if np.sum(next_state_probs) == 0: #at the cul-de-sac
            return (None,
                    -100,
                    True)
        else:
            next_state_idx = np.random.choice(np.arange(next_state_probs.size),p=next_state_probs)
            next_state = self.grid_indices_to_coordinates(next_state_idx)

            return (next_state,
                    self._get_reward(current_state, action),
                    self._terminal_mask[next_state])

    def generate_experience(self, current_state_idx, action_idx):
        sz,sy,sx = self.grid_indices_to_coordinates(current_state_idx)

        next_state_probs = self._T[sz, sy, sx, action_idx, :, :, :].flatten()
        if np.sum(next_state_probs) == 0: #at the cul-de-sac
            return (None,
                    -100,
                    True)

        else:
            next_state_idx = np.random.choice(np.arange(next_state_probs.size),p=next_state_probs)
            if next_state_idx >= 560:
                pdb.set_trace()
            return (next_state_idx,
                    self._get_reward(current_state_idx, action_idx),
                    self._terminal_mask.flatten()[next_state_idx])

    def grid_indices_to_coordinates(self, indices=None):
        if indices is None:
            indices = np.arange(self.size)
        return np.unravel_index(indices, self.shape)

    def grid_coordinates_to_indices(self, coordinates=None):
        # pdb.set_trace()
        if coordinates is None:
            return np.arange(self.size)
        return np.ravel_multi_index(coordinates, self.shape)


    def best_policy(self, utility_grid, discount):
        H, M, N = self.shape
        policy_grid = np.zeros((H,M,N))

        for i in range(M):
            for j in range(N):
                for k in range(H):
                    actions = self._get_actions((k,i,j))
                    values = []
                    for action in actions:
                        reward = self._get_reward((k,i,j),action)
                        possible_next_states = self._get_possible_states((k,i,j))
                        U =0
                        for new_loc in (possible_next_states):
                            z_,y_,x_ = new_loc
                            # pdb.set_trace()
                            U+= self._T[k,i,j,action,z_,y_,x_]*utility_grid[z_,y_,x_]
                        values.append(discount*U+reward)
                    # pdb.set_trace()
                    policy_grid[k,i,j]= actions[np.argmax(values)]
        return policy_grid

        # return np.argmax((utility_grid.reshape((1, 1, 1, 1, H, M, N)) * self._T).sum(axis=-1).sum(axis=-1).sum(axis=-1), axis=3)

    def _init_utility_policy_storage(self, depth):
        H, M, N = self.shape
        utility_grids = np.zeros((H, M, N, depth))
        policy_grids = np.zeros_like(utility_grids)
        return utility_grids, policy_grids

    def _normalize_transition(self,loc, action,T):
        z0, y0, x0 = loc
        # print(T[z0,y0,x0,action,:,:,:] )
        # pdb.set_trace()
        if np.sum(T[z0,y0,x0,action,:,:,:] ) == 0:
            denum = 0.1
        else:
            denum = np.sum(T[z0,y0,x0,action,:,:,:] )
        T[z0,y0,x0,action,:,:,:] = np.divide(T[z0,y0,x0,action,:,:,:] ,denum)


    def _update_transition(self,s, a, a_alt, obstacle_mask,T, P):
        '''
        a_alt: actual action (not necessarily the action you took)
        '''
        H, M, N = self.shape
        z0, y0, x0 = s
        dz, dy, dx = self._direction_deltas[a_alt]
        z1 = np.clip(z0 + dz, 0, H - 1)
        y1 = np.clip(y0 + dy, 0, M - 1)
        x1 = np.clip(x0 + dx, 0, N - 1)

        if not obstacle_mask[z1,y1,x1] :
            T[z0, y0, x0, a, z1, y1, x1] += P
        else:
            T[z0, y0, x0, a, z1, y1, x1] = 0


    def _create_transition_matrix(self,
                                  action_probabilities,
                                  no_action_probability,
                                  obstacle_mask):
        H, M, N = self.shape

        T = np.zeros((H, M, N, self._num_actions, H, M, N))

        z, y, x = self.grid_indices_to_coordinates()

        # up, right, down, left, float, sink, stay
        for action, P in action_probabilities:
            for (z0,(y0,x0)) in zip(z,zip(y,x)):
                self._update_transition((z0, y0, x0), action, action,  obstacle_mask, T, P)
                if action ==0: #wanted to go up
                    self._update_transition((z0, y0, x0), action, 1, obstacle_mask, T, .1)
                    self._update_transition((z0, y0, x0), action, 4, obstacle_mask, T, .05)
                    self._update_transition((z0, y0, x0), action, 5, obstacle_mask, T, .05)
                    self._normalize_transition((z0, y0, x0), action,T)
                    # self._account_for_obstacles((z0,y0,x0), action, (1,4,5),obstacle_mask,T)
                elif action ==1:
                    self._update_transition((z0, y0, x0), action, 0, obstacle_mask, T, .1)
                    self._update_transition((z0, y0, x0), action, 4, obstacle_mask, T, .05)
                    self._update_transition((z0, y0, x0), action, 5, obstacle_mask, T, .05)
                    self._normalize_transition((z0, y0, x0), action,T)
                elif action == 2: #wanted to go down
                    self._update_transition((z0, y0, x0), action, 3, obstacle_mask, T, .2)
                    self._update_transition((z0, y0, x0), action, 0, obstacle_mask, T, .1)
                    self._update_transition((z0, y0, x0), action, 1, obstacle_mask, T, .1)
                    self._normalize_transition((z0, y0, x0), action,T)
                elif action==3:
                    self._update_transition((z0, y0, x0), action, 2, obstacle_mask, T, .2)
                    self._update_transition((z0, y0, x0), action, 0, obstacle_mask, T, .1)
                    self._update_transition((z0, y0, x0), action, 1, obstacle_mask, T, .1)
                    self._normalize_transition((z0, y0, x0), action,T)
                elif action==4: #wanted to float
                    self._update_transition((z0, y0, x0), action, 5, obstacle_mask, T, .1)
                    self._normalize_transition((z0, y0, x0), action,T)
                elif action==5: #wanted to sink
                    self._update_transition((z0, y0, x0), action, 4, obstacle_mask, T, .1)
                    self._normalize_transition((z0, y0, x0), action,T)
                elif action ==6: #wanted to stay
                    self._update_transition((z0, y0, x0), action, 0, obstacle_mask, T, .2)
                    self._update_transition((z0, y0, x0), action, 1, obstacle_mask, T, .2)
                    self._update_transition((z0, y0, x0), action, 2, obstacle_mask, T, .2)
                    self._update_transition((z0, y0, x0), action, 3, obstacle_mask, T, .2)
                    self._normalize_transition((z0, y0, x0), action,T)

        terminal_locs = np.where(self._terminal_mask.flatten())[0]
        T[z[terminal_locs], y[terminal_locs], x[terminal_locs], :, :, :, :] = 0
        # pdb.set_trace()
        return T

    def build_neighbor_dict(self):
        z, y, x = self.grid_indices_to_coordinates()
        H, M, N = self.shape
        self.neighbor_dict = dict()
        for (z0,(y0,x0)) in zip(z,zip(y,x)):
            neighbors=[]
            for delta in self._direction_deltas[:5]:
                neighbor=np.array([z0,y0,x0]) + delta
                if -1 in neighbor:
                    continue
                elif np.any(neighbor-self.shape>=0):
                    continue
                else:
                    neighbors.append(neighbor)
            key = (z0,y0,x0)
            # pdb.set_trace()
            self.neighbor_dict.update({str(key): neighbors})


    def getObservation(self,state):
        z,y,x = state
        o_idx = np.random.choice(np.arange(self._O.shape[0]),p=self._O[:,z,y,x])
        return o_idx

    def similar_landmarks(self,key):
        for i in range(len(self.sets)):
            if str(key) in self.sets[i]:
                return self.sets[i]

    def _create_observation_matrix(self):

        H, M, N = self.shape
        # pdb.set_trace()
        O = np.zeros((self.num_obs, H,M,N))
        z, y, x = self.grid_indices_to_coordinates()
        for (z0,(y0,x0)) in zip(z,zip(y,x)):
            # print('state (%d, %d, %d)'%(z0, y0, x0))
            #returns an array of neighbors that are obstacles. Empty array otherwise (len(returnval)==0
            edges, obstacles = self._find_landmarks((z0,y0,x0))
            key=(obstacles,edges)
            candidates = self.similar_landmarks(key)
            O[self.landmark_lookup[str(key)],z0,y0,x0] +=self.meas_accuracy
            this_candidates = candidates.copy()
            this_candidates.remove(str(key))
            for this_candidate in this_candidates:
                O[self.landmark_lookup[this_candidate],z0,y0,x0] += (1.0-self.meas_accuracy)/len(this_candidates)
                # pdb.set_trace()

        # pdb.set_trace()
        #check
        for (z0,(y0,x0)) in zip(z,zip(y,x)):
            if not np.isclose(np.sum(O[:,z0,y0,x0]),1.0):
                print('obs matrix error')

        return O


    def _value_iteration(self, utility_grid, discount=1.0):

        out = np.zeros_like(utility_grid)
        H, M, N = self.shape
        for i in range(M):
            for j in range(N):
                for k in range(H):
                    if not self._obstacle_mask[k,i,j]:
                        out[k, i, j] = self._update_utility((k, i, j),
                                                         discount,
                                                         utility_grid)
        return out

    def _policy_iteration(self, *, utility_grid,
                          policy_grid, discount=1.0):
        r, c = self.grid_indices_to_coordinates()
        print('policy iteration has not been 3d-fied')

        M, N = self.shape

        utility_grid = (
            self._reward_grid +
            discount * ((utility_grid.reshape((1, 1, 1, M, N)) * self._T)
                        .sum(axis=-1).sum(axis=-1))[r, c, policy_grid.flatten()]
            .reshape(self.shape)
        )

        utility_grid[self._terminal_mask] = self._reward_grid[self._terminal_mask]

        return self.best_policy(utility_grid), utility_grid

    def _get_reward(self, loc, action):
        # up, right, down, left, float, sink, stay
        # pdb.set_trace()
        if type(loc) != tuple:
            loc = self.grid_indices_to_coordinates(loc)
        reward = self._reward_grid[loc]
        # if self._obstacle_mask[loc] == True:
        #     reward -=1        #done in rl.py
        if action <4: # up, right, down, left
            return reward-.5
        elif action == 4: #float
            return reward-2
        elif action == 5:
            return reward-1
        else: #action = stay
            return reward

    def _get_actions(self, loc):
        H,M,N = self._reward_grid.shape
        z,y,x = loc
        actions = []
        if y!=0: #can go up
            if not(self._obstacle_mask[z,y-1,x]):
                actions.append(0)
        if x+1 !=N: #can go right
            if not (self._obstacle_mask[z,y,x+1]):
                actions.append(1)
        if y+1 !=M: #can go down
            if not (self._obstacle_mask[z,y+1,x]):
                actions.append(2)
        if x!=0: #can go left
            if not (self._obstacle_mask[z,y,x-1]):
                actions.append(3)
        if z+1!=H: #can float
            if not (self._obstacle_mask[z+1,y,x]):
                actions.append(4)
        if z !=0: #can sink
            if not (self._obstacle_mask[z-1,y,x]):
                actions.append(5)
        actions.append(6) #can always stay
        return actions

    def _get_possible_states(self,loc):
        z,y,x = loc
        actions = self._get_actions(loc)
        possible_states = []
        for action in actions:
            possible_states.append(self._get_next_state(loc, action))
        # pdb.set_trace()
        return possible_states

    def _get_next_state(self,loc,action):
        return tuple(np.array(loc)+np.array(self._direction_deltas[action]))

    def _update_utility(self, loc, discount, utility_grid):
        z, y, x = loc

        if self._terminal_mask[loc]:
            return self._reward_grid[loc]
        actions = self._get_actions(loc)
        if actions is None:
            print('you should not end up here, since you can always choose STAY action.')
            return None

        newU = []
        for action in actions:
            reward = self._get_reward(loc, action)
            possible_next_states = self._get_possible_states(loc)
            # pdb.set_trace()
            U =0
            for new_loc in possible_next_states:
                z_,y_,x_ = new_loc
                # pdb.set_trace()
                U+= self._T[z,y,x,action,z_,y_,x_]*utility_grid[z_,y_,x_]
            newU.append(discount*U+reward)
        # if z == 1 and y == 1 and x == 0:
            # pdb.set_trace()
        # pdb.set_trace()
        return np.max(newU)

    def _find_landmarks(self,state):
        #given the current state, return the position of the edges and obstacles
        neigh=np.array(state) + self._direction_deltas[:4]
        obstacles=[]; edges =[]
        for i in range(len(neigh)):
            this_neigh = neigh[i,:]
            # print('neighbor:',this_neigh)
            if len(this_neigh[this_neigh<0])>0:
                edges.append(i)
            else:
                try:
                    is_neigh_obstacle = self._obstacle_mask[neigh[i,0],neigh[i,1],neigh[i,2]]
                except:
                    edges.append(i)
                    is_neigh_obstacle = False
                if is_neigh_obstacle:
                    obstacles.append(i)
        if len(edges) == 0:
            edges = None
        elif len(edges) >1:
            edges =tuple(edges)
        else:
            edges = edges[0]

        if len(obstacles) ==0:
            obstacles=None
        elif len(obstacles) >1:
            obstacles =tuple(obstacles)
        else:
            obstacles = obstacles[0]

        return edges, obstacles

    # def _is_on_the_edge(self,state):
    #     '''
    #     returns the index of found edges.
    #     0->z
    #     1->y
    #     2->x
    #     '''
    #     H,M,N = self.shape
    #     #do silly way
    #     state = np.array(state)
    #     found_edges =[]
    #     isOutEdge =  np.divide(np.array(state),np.array([H,M,N]))
    #     for i in range(len(state)):
    #         if np.isclose(isOutEdge[i],1.0) or state[i] == 0:
    #             found_edges.append(i)
    #     return found_edges


    # def _calculate_utility(self, loc, discount, utility_grid):
    #     if self._terminal_mask[loc]:
    #         return self._reward_grid[loc]
    #     z, y, x = loc
    #     actions = self._get_actions(loc)
    #     if actions is None:
    #         print('you should not end up here but....')
    #         return None
    #     Q = np.zeros(actions.shape)
    #     for action in actions:
    #         reward = self._get_reward(loc, action)
    def plot_path(self, path, total_reward, utility_grid):
        # pdb.set_trace()
        path_x = path[:,2]; path_y = path[:,1]; path_z = path[:,0]

        path_mask = np.zeros_like(self._reward_grid)
        path_mask[path_z, path_y, path_x] = True
        H,M,N = self._reward_grid.shape
        # pdb.set_trace()
        markers = "^>v<+|x"  #maybe change this
        marker_size = 200 // np.max(self._reward_grid.shape)
        marker_edge_width = marker_size // 10
        marker_fill_color = 'b'

        no_action_mask = self._terminal_mask | self._obstacle_mask
        umin = utility_grid.min()
        umax = utility_grid.max()
        utility_normalized = (utility_grid - umin ) / \
                             (umax- umin)

        utility_normalized = (255*utility_normalized).astype(np.uint8)

        fig = plt.figure(figsize=(30,10))
        for floor in range(H):
            plotnum = int(str(1)+str(H)+str(floor+1))
            # pdb.set_trace()
            fig.add_subplot(plotnum)

            utility_rgb = cv2.applyColorMap(utility_normalized[floor, :, :], cv2.COLORMAP_BONE)
            for i in range(3):
                channel = utility_rgb[:, :, i]
                channel[self._obstacle_mask[floor,:,:]] = 0

            plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')


            this_path_mask = path_mask[floor,:,:]

            for i, marker in enumerate(markers):
                y, x = np.where((this_path_mask == i) & np.logical_not(no_action_mask[floor,:,:]))
                plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                         color=marker_fill_color)

            y, x = np.where(self._terminal_mask[floor, :,:])
            plt.plot(x, y, '*', ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color)

            if self.start[0] == floor:
                z,y,x = self.start
                plt.plot(x,y, marker=r"$s$",ms = marker_size, mew=marker_edge_width, color='r')

            tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
            tick_step = np.max(path_mask.shape)/8
            best_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
            tick_step = tick_step_options[best_option]
            plt.xticks(np.arange(0, this_path_mask.shape[1] - 0.5, tick_step))
            plt.yticks(np.arange(0, this_path_mask.shape[0] - 0.5, tick_step))
            plt.xlim([-0.5, this_path_mask.shape[0]-0.5])
            plt.xlim([-0.5, this_path_mask.shape[1]-0.5])


    def plot_policy(self, utility_grid, discount, policy_grid=None):
        if policy_grid is None:
            policy_grid = self.best_policy(utility_grid, discount)
        H,M,N = policy_grid.shape
        # pdb.set_trace()
        markers = "^>v<+|x"  #maybe change this
        marker_size = 200 // np.max(policy_grid.shape)
        marker_edge_width = marker_size // 10
        marker_fill_color = 'b'

        no_action_mask = self._terminal_mask | self._obstacle_mask
        umin = utility_grid.min()
        umax = utility_grid.max()
        utility_normalized = (utility_grid - umin ) / \
                             (umax- umin)

        utility_normalized = (255*utility_normalized).astype(np.uint8)


        # pdb.set_trace()
        fig = plt.figure(figsize=(30,10))
        for floor in range(H):
            plotnum = int(str(1)+str(H)+str(floor+1))
            # pdb.set_trace()
            fig.add_subplot(plotnum)

            utility_rgb = cv2.applyColorMap(utility_normalized[floor, :, :], cv2.COLORMAP_BONE)
            for i in range(3):
                channel = utility_rgb[:, :, i]
                channel[self._obstacle_mask[floor,:,:]] = 0

            plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')


            this_policy_grid = policy_grid[floor,:,:]

            for i, marker in enumerate(markers):
                y, x = np.where((this_policy_grid == i) & np.logical_not(no_action_mask[floor,:,:]))
                # print('marking (x,y)=')
                # print(x)
                # print(y)
                plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                         color=marker_fill_color)

            y, x = np.where(self._terminal_mask[floor, :,:])
            plt.plot(x, y, '*', ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color)

            if self.start[0] == floor:
                z,y,x = self.start
                plt.plot(x,y, marker=r"$s$",ms = marker_size, mew=marker_edge_width, color='w')

            # tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
            # tick_step = np.max(policy_grid.shape)/8
            # best_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
            # tick_step = tick_step_options[best_option]
            # plt.xticks(np.arange(0, this_policy_grid.shape[1] - 0.5, tick_step))
            # plt.yticks(np.arange(0, this_policy_grid.shape[0] - 0.5, tick_step))
            plt.xlim([-0.5, this_policy_grid.shape[0]-0.5])
            plt.xlim([-0.5, this_policy_grid.shape[1]-0.5])
        # pdb.set_trace()
        # plt.suptitle('U_min= %.2f, U_max = %.2f' %(umin, umax))
