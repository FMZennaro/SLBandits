
import numpy as np

class GaussianBandit():
    def __init__(self,k):   
        self.k = k
    
    def initialize_gaussian_reward_distributions(self,mu,sigma):    
        self.means = np.random.normal(mu,sigma,self.k)
        self.variances = np.ones(self.k)
        
    def initialize_fixed_reward_distributions(self,means,variances):
        self.means = means
        self.variances = variances
        
    def get_optimal_action(self):
        return np.argmax(self.means)    
        
    def reset(self):
        return None, 0, False, None
    
    def step(self,action):        
        reward = np.random.normal(self.means[action],self.variances[action])
        return None, reward, True, None

class Bandit():
    
    def __init__(self,k,mu,sigma):   
        self.k = k
        self.means = np.random.normal(mu,sigma,k)
        self.variances = np.ones(k)
        
    def get_optimal_action(self):
        return np.argmax(self.means)    
        
    def reset(self):
        return None, 0, False, None
    
    def step(self,action):        
        reward = np.random.normal(self.means[action],self.variances[action])
        return None, reward, True, None