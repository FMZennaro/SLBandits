
import numpy as np
import scipy.stats as stats

def recast_p_for_scipy(p):
    p[p<1e-14] = 1e-14
    #p = np.asarray(p,dtype='float64')
    p = p / np.sum(p)
    return p


class Agent():
    def __init__(self,env,k):
        self.env = env
        self.n_actions = k
        self.steps = np.zeros(self.n_actions)
        self.rewards = np.zeros(self.n_actions)
    
    def initialize_Q_zeros(self):
        self.Q = np.zeros(self.n_actions)      
    
    def initialize_Q_ones(self):
        self.Q = np.ones(self.n_actions)
        
    def initialize_Q_c(self,c):
        self.Q = np.ones(self.n_actions)*c
    
    def get_epistemic_uncertainty(self):
        return np.nan
    
    def get_total_uncertainty(self):
        return np.nan
        
    def run(self):
        action = self._select_action()
        reward = self.step(action)
        return action,reward
        
    def step(self,action):
        self.steps[action] = self.steps[action]+1
        _,reward,_,_ = self.env.step(action)
        self.rewards[action] = self.rewards[action] + reward
        self._update_agent(action,reward)
        return reward
    
class Agent_epsilon(Agent):
    
    def __init__(self,env,n_actions,eps):
        super().__init__(env,n_actions)
        self.eps = eps
           
    def _select_action(self):
        if(np.random.random() < self.eps):
            return np.random.randint(0,self.n_actions)
        else:
            return np.argmax(self.Q)
        
    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + (1./self.steps[action])*(reward-self.Q[action])
        
class Agent_epsilondecay(Agent_epsilon):
    
    def __init__(self,env,n_actions,decay):
        super().__init__(env,n_actions,None)
        self.decay = decay
           
    def _select_action(self):
        threshold = 1. / np.sum(self.steps)**(1./self.decay)
        if(np.random.random() < threshold):
            return np.random.randint(0,self.n_actions)
        else:
            return np.argmax(self.Q)
        
class Agent_UCB(Agent):  
    
    def __init__(self,env,n_actions,c):
        super().__init__(env,n_actions)
        self.c = c
           
    def _select_action(self):        
        a = self.Q + self.c * np.sqrt( np.log(np.sum(self.steps)) / self.steps  )
        return np.argmax(a)
        
    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + (1./self.steps[action])*(reward-self.Q[action])
        
class Agent_UCB1Normal(Agent_UCB):  
    
    def __init__(self,env,n_actions,c):
        super().__init__(env,n_actions,c)
        self.squaredrewards = np.zeros(self.n_actions)
           
    def _select_action(self):
        #threshold = np.ceil(np.log(np.sum(self.steps)))
        threshold = 0
        if np.any(self.steps < threshold):
            return np.random.choice(np.where(self.steps < threshold)[0])
        else:            
            a = self.Q + self.c * np.sqrt((self.squaredrewards - self.steps*self.Q**2)/(self.steps-1) * 
                                          (np.log(np.sum(self.steps)-1))/(self.steps) )
            return np.argmax(a)
        
    def step(self,action):
        self.steps[action] = self.steps[action]+1
        _,reward,_,_ = self.env.step(action)
        self.rewards[action] = self.rewards[action] + reward
        self.squaredrewards[action] = self.squaredrewards[action] + reward**2
        self._update_agent(action,reward)
        return reward

class Agent_GradientBoltzmann(Agent):

    def __init__(self,env,n_actions,eta):
        super().__init__(env,n_actions)
        self.eta = eta
        
    def initialize_H_zeros(self):
        self.H = np.zeros(self.n_actions)
        
    def get_epistemic_uncertainty(self):
        return np.nan
    
    def get_total_uncertainty(self):
        ps = self._compute_action_distribution()
        return stats.multinomial.entropy(1, ps)
    
    def _compute_action_distribution(self):
        return np.exp(self.H) / np.sum(np.exp(self.H))
           
    def _select_action(self): 
        pi = self._compute_action_distribution()      
        action = np.random.multinomial(1,pi)
        return np.argmax(action)
        
    def _update_agent(self,action,reward):
        pi = self._compute_action_distribution()
        m = self.eta * (reward - np.sum(self.rewards)/np.sum(self.steps))
        
        self.H = self.H - m * pi
        self.H[action] = self.H[action] + m*pi[action] + m * (1-pi[action])


class Agent_Subjective(Agent):
    
    def __init__(self,env,n_actions,eta=1.):
        self.eta = eta
        super().__init__(env,n_actions)
        
    def initialize_empty_opinion(self):
        self.b = np.zeros(self.n_actions)
        self.u = 1.
        self.c = np.ones(self.n_actions) / self.n_actions
        
        self.W = 2
    
    def get_epistemic_uncertainty(self):
        return self.u
    
    def get_total_uncertainty(self):
        ps = self._compute_action_distribution()
        return stats.multinomial.entropy(1, ps)
    
    def _compute_action_distribution(self):
        ps = self.b + self.u*self.c
        ps = recast_p_for_scipy(ps)
        return ps
           
    def _select_action(self):
        pi = self._compute_action_distribution()        
        action = np.random.choice(self.n_actions,1,p=pi)[0]
        return action
        
    def _update_agent(self,action,reward):
        raise NotImplemented
        
class Agent_SubjectiveEvidential(Agent_Subjective):
        
    def _update_agent(self,action,reward):
        r = (self.W * self.b) / self.u
        r[action] = r[action] + 1
        
        self.b = r / (self.W + np.sum(r))
        self.u = self.W / (self.W + np.sum(r))
        
class Agent_SubjectiveEvidential_avgreward(Agent_Subjective):
        
    def _update_agent(self,action,reward):
        r = (self.W * self.b) / self.u
        
        if (reward >= np.mean(self.rewards/self.steps)):
            r[action] = r[action] + self.eta
        
            self.b = r / (self.W + np.sum(r))
            self.u = self.W / (self.W + np.sum(r))
            
class Agent_SubjectiveEvidential_maxreward(Agent_Subjective):
        
    def _update_agent(self,action,reward):
        r = (self.W * self.b) / self.u
        
        if (reward >= np.max(self.rewards/self.steps)):
            r[action] = r[action] + self.eta
        
            self.b = r / (self.W + np.sum(r))
            self.u = self.W / (self.W + np.sum(r))

class Agent_SubjectiveEvidential_maxrewardscaled(Agent_Subjective):
        
    def _update_agent(self,action,reward):
        r = (self.W * self.b) / self.u
        
        if (reward >= np.max(self.rewards/self.steps)):
            r[action] = r[action] + self.eta*(reward - np.max(self.rewards/self.steps))
        
            self.b = r / (self.W + np.sum(r))
            self.u = self.W / (self.W + np.sum(r))
            
class Agent_SubjectiveEvidential_maxreward2scaled(Agent_Subjective):
        
    def _update_agent(self,action,reward):
        r = (self.W * self.b) / self.u
        
        estimated_rewards = self.rewards/self.steps
        bestaction = np.argmax(estimated_rewards)
        estimated_rewards[bestaction] = -np.inf
        secondbestaction = np.argmax(estimated_rewards)
        estimated_rewards[bestaction] = self.rewards[bestaction]/self.steps[bestaction]
        
        if(action==bestaction):
            if(reward > estimated_rewards[secondbestaction]):
                r[action] = r[action] + self.eta*(reward - estimated_rewards[secondbestaction])
        
        elif(reward >= estimated_rewards[bestaction]):
            r[action] = r[action] + self.eta*(reward - estimated_rewards[bestaction])
        
        self.b = r / (self.W + np.sum(r))
        self.u = self.W / (self.W + np.sum(r))
            
           
class Agent_SubjectiveEvidential_upsilon(Agent_SubjectiveEvidential_maxrewardscaled):
               
    def _select_action(self):
        pi = self._compute_action_distribution()  
        if(np.random.random() < self.u):      
            return np.random.choice(self.n_actions,1,p=pi)[0]
        else:
            return np.argmax(pi)
        
class Agent_SubjectiveEvidential_upsilon2(Agent_SubjectiveEvidential_maxrewardscaled):
               
    def _select_action(self):
        pi = self._compute_action_distribution()  
        if(np.random.random() < self.u):      
            return np.random.choice(self.n_actions,1,p=pi)[0]
        else:
            return np.argmax(pi)
        
    def _update_agent(self,action,reward):
        r = (self.W * self.b) / self.u
        
        if (reward >= np.max(self.rewards/self.steps)):
            r[action] = r[action] + (1 / self.u)*(reward - np.max(self.rewards/self.steps))
        
            self.b = r / (self.W + np.sum(r))
            self.u = self.W / (self.W + np.sum(r))
