import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T

#from underfit_choice import *
from bayesian_choice import *
from predictiveness import *
from get_moments import get_predictiveness_array, get_moments
from model_choice_util import load_models, load_obs, make_agression_profiles, make_priors_profiles, kData, kPred

import time
import cPickle as pickle

    
#INFERENCE = 'underfit'
INFERENCE = 'bayes'
BATCH_SIZE = 10000


if INFERENCE == 'underfit':
    pValue_alg = 1
    num_alpha = 10
    num_profiles = 3 # profiles = agression functions 
    name = 'underfit_nA%d_pv%d'%(num_alpha, pValue_alg)
    
elif INFERENCE == 'bayes':
    num_priors = 1
    num_loss_funcs = 4
    num_profiles = num_priors*num_loss_funcs # profiles = prior + loss pair
    name = 'bayes'


def model_choice(models, obs):	
    k = [i for i in xrange(2, 9)]
    
    Statistics = []
    
    for ki in k:        
        print 'K = ', ki
                
        num_M = models[ki-2].shape[0]
        
        numNbins = len(obs[ki-2])
        numHbins = len(obs[ki-2][0])
                
        M = theano.shared(np.asarray(models[ki-2], dtype = theano.config.floatX))
           
        ObSym = T.matrix() # Symbolic tensor for observation batches - indexed elements of Obs shared variable are passed through this

        Pred = theano.function([], predictiveness_profiles(M, ki, len(models[ki-2])))() # This should be dealt with better too...

        # setup inference schemas and theano symbolic tensors
        if INFERENCE == 'underfit':
            profiles = make_agression_profiles(num_profiles, num_alpha)
            
            alpha = theano.shared(np.asmatrix(np.linspace(0.0,1.0, num = num_alpha, endpoint = False), dtype=theano.config.floatX))
            Agression_profiles = T.matrix('Agr')
            nAlpha, nM, nO = T.iscalars('','','')

            Choice_Profile = call_underfit_choice_theano(M, Obs, num_M, n_obs, ki, num_alpha, profiles, Pred) # this may never work for the direct pvalues
            #makeChoice = thean.function ...
            
            
        elif INFERENCE == 'bayes':
            profiles = make_priors_profiles(num_priors, num_M)
            
            Priors_profiles = T.matrix('Priors')
            Loss_funcs = T.arange(1,5)  # Loss functions are choices in bayesian_choice numbered [1,4]
            nM, nO = T.iscalars('','')

            Choice_Maker = Bayesian_Choice(M, ObSym, nM, nO, ki, Priors_profiles, Loss_funcs)
            
        else:
            print 'unknown inference algorithm...'
            quit()
        
        
        
        
                    

        
        # all data for this K
        k_Data = kData(numNbins, numHbins, num_profiles)
        
        
        
        for i in xrange(numNbins):
            for j in xrange(numHbins):
                print 'bin ', i, j
                
                t0 = time.time()
                
                if obs[ki-2][i][j] == [] or obs[ki-2][i][j][0].shape[1] == 0:
                    #there are no observtions in this N*H bin...
                    continue
                else:
                    num_obs = obs[ki-2][i][j][0].shape[0]

                # allocate for predictiveness of model choice vs universe for each obs for each profile
                k_pred = kPred(num_obs, num_profiles)

                num_batches = int(np.ceil(num_obs/np.float(BATCH_SIZE)))

                for batch_index in xrange(num_batches):
                    top = BATCH_SIZE*(batch_index+1) if batch_index < (num_batches-1) else num_obs
                    
                    print 'batch index ', batch_index, '\t num obs: ', num_obs
                    
                    if INFERENCE == 'underfit':                        
                        for prof in xrange(num_profiles):
                            k_pred[prof][BATCH_SIZE*(batch_index):top] = get_predictiveness_array(batch_choice[prof],  obs[ki-2][i][j][1], Pred, n_obs)

                    elif INFERENCE == 'bayes':   
                        batch_choice = Choice_Maker.Choice_Profile_F(profiles, num_M, num_obs, obs[ki-2][i][j][0])
                        print batch_choice
                     
                        for pr in xrange(num_priors):
                            for lf in xrange(num_loss_funcs):
                                k_pred[pr*num_loss_funcs + lf][BATCH_SIZE*(batch_index):top] = get_predictiveness_array(batch_choice[pr][lf],  obs[ki-2][i][j][1], Pred, num_obs)
                        
                for prof in xrange(num_profiles):
                    pred_moments = get_moments(k_pred[prof], num_obs)
                    for m in xrange(len(pred_moments)):
                        k_Data[prof][m][i,j] = pred_moments[m]

                t1 = time.time()
                print 'single bin takes: ',(t1-t0)/60., ' minutes' 
        Statistics.append(k_Data)
        f = open('%s_k%d.pkl'%(name, ki), 'wb')
        pickle.dump(k_Data, f)
        f.close()
    
    return Statistics


models = load_models()
obs = load_obs()
stats = model_choice(models, obs)

