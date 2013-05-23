import numpy as np
import theano
import time
import cPickle as pickle


def load_models(filename = 'models.pkl'):
    t0 = time.time()
    f = open(filename, 'rb')
    models = pickle.load(f)
    f.close()
    t1 =time.time()
    print t1-t0

    return models

def load_obs(filename = 'obs10000.pkl'):
    t0 = time.time()
    f = open(filename, 'rb')
    obs = pickle.load(f)
    f.close()
    t1 =time.time()
    print t1-t0
    
    return obs

def make_agression_profiles(num_agression_profiles, num_alpha):
    Agression_profiles = np.zeros((num_agression_profiles, num_alpha))
    Agression_profiles[0] = np.ones((num_alpha), dtype = theano.config.floatX)/num_alpha #uniform
    Agression_profiles[1] = np.array([i*2./num_alpha for i in xrange(num_alpha)], dtype = theano.config.floatX) #agressive
    Agression_profiles[2] = np.array([2-i*2./num_alpha for i in xrange(num_alpha)], dtype = theano.config.floatX) #cautious
    
    return Agression_profiles

def make_priors_profiles(num_profiles, num_models):
    Priors_profiles = np.zeros((num_profiles, num_models))
    Priors_profiles[0] = np.ones((num_models))
    #Priors_profiles[1] = np.arange(num_models, num_models*2)#np.ones((num_models))
    
    return Priors_profiles

def kData(numNbins, numHbins, num_profiles):
        # mean, variance, kurtosis matrices.         #for each profile
        k_Data = [[np.zeros((numNbins, numHbins)), np.zeros((numNbins, numHbins)), np.zeros((numNbins, numHbins))]]
        k_Data = k_Data*num_profiles
        
        return k_Data

def kPred(num_obs, num_profiles):
    k_pred = [[np.zeros((num_obs))]]
    k_pred = k_pred*num_profiles

    return k_pred