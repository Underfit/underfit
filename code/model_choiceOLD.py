import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T

from underfit_choice import call_underfit_choice_theano
from predictiveness import *
from get_moments import get_predictiveness_array, get_moments

import time
import cPickle as pickle


t0 = time.time()
f = open('models.pkl', 'rb')
models = pickle.load(f)
f.close()
t1 =time.time()
print t1-t0
t0 = time.time()

f = open('obs10000.pkl', 'rb')
obs = pickle.load(f)
f.close()

t1 =time.time()
print t1-t0





def model_choice(models, obs):	
    k = [i for i in xrange(2, 9)]
    num_alpha = 10
    
    Choices = []
    Statistics = []

    Agression_profiles = []
    Agression_profiles.append(theano.shared(np.ones((num_alpha, 1), dtype = theano.config.floatX)/num_alpha)) #uniform
    Agression_profiles.append(theano.shared(np.array([[i*2./num_alpha] for i in xrange(num_alpha)], dtype = theano.config.floatX))) #agressive
    Agression_profiles.append(theano.shared(np.array([[2-i*2./num_alpha] for i in xrange(num_alpha)], dtype = theano.config.floatX))) #cautious
    
    for ki in k:
        Choices.append([])
        Statistics.append([])
        
        print 'K = ', ki
        
        #Agression = theano.shared(np.ones((num_alpha, 1), dtype = theano.config.floatX)/num_alpha)
        
        num_M = models[ki-2].shape[0]
        
        M = theano.shared(np.asarray(models[ki-2], dtype = theano.config.floatX))
        
        result = predictiveness_profiles(M, ki, len(models[ki-2]))
        
        t0=time.time()
        Pred = theano.function([], result)()
        t1=time.time()
        print 'pred took ', t1-t0, ' s'
        numNbins = len(obs[ki-2])
        numHbins = len(obs[ki-2][0])


        for i in xrange(numNbins):
            #Choices[ki-2].append([])
            Statistics[ki-2].append([])

            for j in xrange(numHbins):
                print 'bin ', i, j
                
                t0 = time.time()
                
                if obs[ki-2][i][j] == [] or obs[ki-2][i][j][0].shape[1] == 0:
                    #there are no observtions in this N*H bin...
                    #Choices[ki-2][i].append([])
                    Statistics[ki-2][i].append([])
                    continue
                else:
                    num_obs = obs[ki-2][i][j][0].shape[0]
                print 'num obs ', num_obs

                if num_obs < 1000:
                    Obs = theano.shared(np.asarray(obs[ki-2][i][j][0], dtype = theano.config.floatX))
                    choice_bin = call_underfit_choice_theano(M, Obs, num_M, num_obs, ki, num_alpha, Agression_profiles, Pred)
                else:
                    Obs = theano.shared(np.asarray(obs[ki-2][i][j][0][0:1000], dtype = theano.config.floatX))
                    n_obs = len(Obs.get_value())
                    print n_obs
                    choice_bin = call_underfit_choice_theano(M, Obs, num_M, n_obs, ki, num_alpha, Agression_profiles, Pred) 
                    for batch_index in xrange(int(np.floor(num_obs/1000))):
                        top = 1000*(batch_index+2) if batch_index < (int(np.floor(num_obs/1000))-1) else len(obs[ki-2][i][j][0])
                        Obs = theano.shared(np.asarray(obs[ki-2][i][j][0][1000*(batch_index+1):top]))
                        n_obs = len(Obs.get_value())
                        print n_obs
                        batch_choice = call_underfit_choice_theano(M, Obs, num_M, n_obs, ki, num_alpha, Agression_profiles, Pred)
                        for ag in xrange(len(Agression_profiles)):
                            choice_bin[ag] = np.asarray(choice_bin[ag].tolist() + batch_choice[ag].tolist()) #appends batches for each agression function
                t1=time.time()
            
                print choice_bin#, obs[ki-2][i][j][1]
                
                print 'choices took ', t1-t0, ' s'
                
                #Choices[ki-2][i].append([choice_bin, obs[ki-2][i][j][1]])

                agg_bin = []
                for a in xrange(len(Agression_profiles)):
                    choice_pred = get_predictiveness_array(choices = choice_bin[a], verses = obs[ki-2][i][j][1], Preds = Pred, num_Obs = num_obs)
                    pred_moments = get_moments(choice_pred, num_obs)
                    agg_bin.append(pred_moments)
                
                Statistics[ki-2][i].append(agg_bin)

                t1 = time.time()


                print 'single bin takes: ',(t1-t0)/60., ' minutes' 
    
                #print choice_bin
            f = open('k%d_%d_chi_statistics.pkl'%(ki, i), 'wb')
            pickle.dump(Statistics[ki-2][i], f)
            f.close()
        f = open('k%d_chi_statistics.pkl'%ki, 'wb')
        pickle.dump(Statistics[ki-2],f)
        f.close()
    return Choices

choices = model_choice(models, obs)
#f = open('choices.pkl', 'wb')
#pickle.dump(choices, f)
#f.close()
