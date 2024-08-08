import numpy as np

class BufferReplay():

    def __init__(self, 
                 max_memory_size = int(1e6), 
                 state_size      = 128, 
                 N_action        = 4, 
                 N_action_type   = 3,
                 alpha           = 0.6):
      
        self.max_memory_size = max_memory_size
        self.memory_cntr     = 0
        self.state_size      = state_size
        self.N_action        = N_action
        self.N_action_type   = N_action_type

        #current state
        self.s      = np.zeros((self.max_memory_size, self.state_size))
        #current action
        self.a      = np.zeros((self.max_memory_size, self.N_action))
        #current action type
        self.a_type = np.zeros((self.max_memory_size, self.N_action_type))
        #next reward
        self.r      = np.zeros(self.max_memory_size)
        #next state
        self.ns     = np.zeros((self.max_memory_size, self.state_size))
        #is done in the next state
        self.done   = np.zeros(self.max_memory_size, dtype = bool)
        #predicted q value: (q1 + q2)/2.
        self.predict_q = np.zeros(self.max_memory_size)
        #labeled q value: reward + gamma*min(target_q1, target_q2)
        self.labeled_q = np.zeros(self.max_memory_size)
        #surprise value
        self.priority  = np.ones(self.max_memory_size)
        #initialise if the memory is full
        self.is_full   = False
        #initialise power value for prioritisation
        self.alpha     = alpha
        #initialise small constant to prevent division by zero
        self.sm_c      = 1e-6

    def store_transition(self, s, a, a_type, r, ns, done, predict_q, labeled_q):

        #update memory
        if self.memory_cntr >= self.max_memory_size:
            self.is_full = True
            self.memory_cntr = 0

        self.s[self.memory_cntr]      = s
        self.a[self.memory_cntr]      = a
        self.a_type[self.memory_cntr] = a_type
        self.r[self.memory_cntr]      = r
        self.ns[self.memory_cntr]     = ns
        self.done[self.memory_cntr]   = done
        self.predict_q[self.memory_cntr] = predict_q
        self.labeled_q[self.memory_cntr] = labeled_q
        self.priority[self.memory_cntr]  = np.abs(predict_q - labeled_q + self.sm_c)**self.alpha

        #update memory counter
        self.memory_cntr += 1

    def sample_buffer(self, batch_size):
        
        #get max_ind for sampling range
        max_ind = self.max_memory_size if self.is_full else self.memory_cntr

        #get priorities
        priorities = self.priority[:max_ind]
        if priorities.sum() == 0:
            priorities = np.ones_like(priorities)
        probs = priorities/(priorities.sum())

        batch   = np.random.choice(max_ind, 
                                   batch_size,
                                   replace = False, 
                                   p = probs)

        s      = self.s[batch]
        a      = self.a[batch]
        a_type = self.a_type[batch]
        r      = self.r[batch]
        ns     = self.ns[batch]
        done   = self.done[batch]

        return batch, s, a, a_type, r, ns, done
    
    def update_buffer(self, sample_ind, predict_q):
        
        self.predict_q[sample_ind] = predict_q
        self.priority[sample_ind]  = np.abs(self.predict_q[sample_ind] - self.labeled_q[sample_ind] + self.sm_c)**self.alpha