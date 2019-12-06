import numpy as np
import random as rand
import pdb

class POMDPSolver:
    def __init__(self,gw,
                 state_shape,
                 start_positions,
                 num_actions,
                 num_obs,
                 depth=2,
                 gamma = 1
                 ):
        self.shape = state_shape
        self.start_positions = start_positions
        self._num_states = self.shape[0] * self.shape[1] * self.shape[2]
        self._num_actions = num_actions
        self._num_obs = num_obs
        self.eps = .001
        self.gw = gw
        self.d = depth
        self.gamma = gamma

    def init_belief(self):
        b0 = np.zeros(self.shape)
        for start_candidate in self.start_positions:
            b0[start_candidate]=1.0/len(self.start_positions)
        return b0


    def update_belief(self,o):
        #update belief for the entire state based on the observation.
        print('update')
        z, y, x = self.gw.grid_indices_to_coordinates()
        new_belief = np.zeros_like(self.b)
        for (z0,(y0,x0)) in zip(z,zip(y,x)): #for all current state
            key = (z0,y0,x0)
            neighbors = self.gw.neighbor_dict[str(key)]
            tb= 0
            # pdb.set_trace()
            for neighbor in neighbors: #sbar
                zbar, ybar, xbar = neighbor
                for action in range(self._num_actions):
                    tb+=self.gw._T[z0,y0,x0,action,zbar,ybar,xbar] * self.b[z0,y0,x0]
                tb*=self.gw._O[o,z0,y0,x0]
            new_belief[z0,y0,x0] = tb
        new_belief = np.divide(new_belief,np.sum(new_belief))
        return new_belief

    def get_reward(b,a):
        print('fill me in')
        return 0

    #how to init alpha vectors/
    def qmdp(self):
        #eqn 6.26
        new_alpha = 0
        return new_alpha

    def getU(self,b):
        print('fill me in')
        #eqn 6.24
        return 1

    def obs_prob(self,b,a):
        #∑_s P(o,|s,a)b(s)  
        #return P(o|b,a)
        return 1

    def selectAction(self,b,d):
        if d == 0:
            return (None, self.getU(b))
        best_action, best_utility = (None, -flaot('inf'))
        for a in range(self._num_actions)):
            current_utility = self.get_reward(bp,a)
            for o in range(self._num_obs):
                bp= update_beleif(o)
                new_a, new_u = selectAction(bp,d-1)
                current_utility = current_utiltiy + self.gamma * self.obs_prob(b,a) * new_u

            if current_utility > best_utility:
                best_action = a
                best_utility = current_utility

            return best_action, best_utility
            #how am i updating my belief?



    def learn_fwsearch(self,U):
        #each individual experience
        b = self.init_belief()
        true_state = self.gw.start
        while True:
            obs_idx = self.gw.getObservation(true_state)
            # obs = self.gw.landmark_idx_lookup[obs_idx]
            self.update_belief(obs_idx)
            pdb.set_trace()
            action, utility = self.selectAction(b,self.d) #twostep lookahead
            next_state, reward, done =generate_experience_with_grid(true_state,action)
            if done:
                print('do more stuff')
                break

        pass

    # def find_alpha_vector(self):
        # pass
    def solve(self,max_iters=100):
        iter = 0
        U = 0
        while iter < max_iters:
            print('iter %d'%iter)
            self.learn_fwsearch(U)
            iter +=1


        pdb.set_trace()
