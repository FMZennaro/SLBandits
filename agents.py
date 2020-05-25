
import numpy as np
import scipy.stats as stats

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

class Agent_optimistic(Agent_epsilon):
    
    def __init__(self,env,n_actions,eps):
        super().__init__(env,n_actions,eps)
    
    def initialize_Q_optimistic(self,q):
        super().initialize_Q_c(q)

class Agent_nonstationary(Agent_optimistic):
    
    def __init__(self,env,n_actions,eps,eta):
        super().__init__(env,n_actions,eps)
        self.eta = eta
        
    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + self.eta*(reward-self.Q[action])
        
class Agent_UCB(Agent):  
    
    def __init__(self,env,n_actions,c):
        super().__init__(env,n_actions)
        self.c = c
           
    def _select_action(self):        
        a = self.Q + self.c * np.sqrt( np.log(np.sum(self.steps)) / self.steps  )
        return np.argmax(a)
        
    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + (1./self.steps[action])*(reward-self.Q[action])
        
class Agent_UCB_nonstationary(Agent_UCB):  
    
    def __init__(self,env,n_actions,c,eta):
        super().__init__(env,env,n_actions,c)
        self.eta = eta

    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + self.eta*(reward-self.Q[action])
        
class Agent_GradientBoltzmann(Agent):

    def __init__(self,env,n_actions,eta):
        super().__init__(env,n_actions)
        self.eta = eta
        
    def initialize_H_zeros(self):
        self.H = np.zeros(self.n_actions) 
    
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
        
class Agent_GradientBoltzmann_fixedbaseline(Agent_GradientBoltzmann):

    def __init__(self,env,n_actions,eta,baseline):
        super().__init__(env,n_actions,eta)
        self.baseline = baseline
        
    def _update_agent(self,action,reward):
        pi = self._compute_action_distribution()
        m = self.eta * (reward - self.baseline)
        
        self.H = self.H - m * pi
        self.H[action] = self.H[action] + m*pi[action] + m * (1-pi[action])


#Agent_softmaxPreference = Agent_Boltzmann
#Agent_softmaxPreference_fixedbaseline = Agent_Boltzmann_fixedbaseline


class Agent_ConjugateGradientBoltzmann(Agent):

    def __init__(self,env,n_actions,eta):
        super().__init__(env,n_actions)
        self.eta = eta
        
    def initialize_H_zeros(self):
        self.H = np.zeros(self.n_actions)
        
    def get_epistemic_uncertainty(self):
        alphas = self._compute_action_distribution()
        return stats.dirichlet.entropy(alphas)
    
    def get_total_uncertainty(self):
        alphas = self._compute_action_distribution()
        ps = alphas / np.sum(alphas)
        return stats.multinomial.entropy(1, ps)
    
    def _compute_action_distribution(self):
        alphas = np.exp(self.H) / np.sum(np.exp(self.H))
        return alphas
           
    def _select_action(self): 
        alphas = self._compute_action_distribution()
        self.pi = stats.dirichlet.rvs(alphas, 1)[0]
        self.pi[self.pi<1e-14] = 1e-14
        self.pi = np.asarray(self.pi,dtype='float64')
        self.pi = self.pi / np.sum(self.pi)
        action = stats.multinomial.rvs(1, self.pi, 1)[0]
        return np.argmax(action)
        
    def _update_agent(self,action,reward):
        m = self.eta * (reward - np.sum(self.rewards)/np.sum(self.steps))
        
        self.H = self.H - m * self.pi
        self.H[action] = self.H[action] + m*self.pi[action] + m * (1-self.pi[action])
        
class Agent_ConjugateGradientBoltzmann_fixedbaseline(Agent_ConjugateGradientBoltzmann):

    def __init__(self,env,n_actions,eta,baseline):
        super().__init__(env,n_actions,eta)
        self.baseline = baseline
        
    def _update_agent(self,action,reward):
        m = self.eta * (reward - self.baseline)
        
        self.H = self.H - m * self.pi
        self.H[action] = self.H[action] + m*self.pi[action] + m * (1-self.pi[action])


class Agent_Subjective(Agent):
    
    def __init__(self,env,n_actions):
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
        return ps
           
    def _select_action(self):
        pi = self._compute_action_distribution()
        pi[pi<1e-14] = 1e-14
        pi = np.asarray(pi,dtype='float64')
        pi = pi / np.sum(pi)
        action = stats.multinomial.rvs(1, pi, 1)[0]
        return np.argmax(action)
        
    def _update_agent(self,action,reward):
        raise NotImplemented
        
class Agent_SubjectiveEvidential(Agent_Subjective):
        
    def _update_agent(self,action,reward):
        r = (self.W * self.b) / self.u
        r[action] = r[action] + 1
        
        self.b = r / (self.W + np.sum(r))
        self.u = self.W / (self.W + np.sum(r)) 

    
class Agent_SubjectiveGradient(Agent_Subjective):
    def __init__(self,env,n_actions,eta):
        self.eta = eta
        super().__init__(env,n_actions)
   
    def _update_agent(self,action,reward):
        m = self.eta * (reward - np.sum(self.rewards)/np.sum(self.steps))
        
        self.b = self.b - m * self.c
        self.b[action] = self.b[action] + m * self.c[action] + m * (1 - self.c[action])
        self.u = 1 - np.sum(self.b)
