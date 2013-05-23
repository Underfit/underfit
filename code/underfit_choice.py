import numpy as np
import theano
import theano.tensor as T
from predictiveness import *
try:
    from theano.sandbox.linalg.kron import kron
except:
    from theano.toolbox import kron

from chi2pvalue import chi2_pvalues
from directpvalue import direct_pvalues

import time
#For fixing log(0) = -INF
INF = 100

#Agression is num_alpha by 1.. it serves as the measure on the unit interval (broken into num_alpha parts)
#Models is num_M by K
#Obs is num_Obs by K
#pValue_alg (0 or 1) toggles between two pvalue algorithms:
# 0 - a Chi-2 approximation
# 1 - a direct exact calculation
def underfit_choice(Agression, num_alpha, Models, Obs, K, num_M, num_Obs, pvalues, alpha, Preds = None):
    #Switch for pvalue algorithm.. first is fast and an approximation, second slow and exact
    
    if Preds == None:
        Preds = predictiveness_profiles(Models, K, num_M)    
    

    # Letting P denote the whole predictiveness matrix, we will tile num_alpha*num_Obs of them together as such:
    #   PPPPPPPPPPPPPPPPPP..
    #Then the pvalues will be broadcastand tiled so that if we number blocks of pvalues their ordering will be
    #   123412341234,, with each block having the pvalues of generating models behind the predictiveness profiles 
    #Then alpha will be tiled so that there's a new alpha for each time the pvalues repeat:
    #111111222222333333...  The idea is to get every triple (alpha, observation, P)
    #
    #This visualization might help:
    #
    #PPPPPPPPPPPP
    #123412341234
    #111122223333
    #
    #Finally the code:
    
    Preds = kron(T.ones((1,num_alpha*num_Obs)),Preds)
    pvalues = kron(pvalues,T.ones((1,num_M)))
    pvalues = kron(T.ones((1,num_alpha)),pvalues)
    alpha = kron(alpha,T.ones((num_M,num_M*num_Obs)))
    
    
    # This "knocks out" rejected universes from predictiveness profiles
    Scores = Preds + INF*(pvalues < alpha) 
    # The worst case predictiveness for each alpha is found
    Scores = T.min(Scores, axis = 0)
    # Rearranging to put vary the alpha between the rows 
    Scores = T.reshape(Scores,(num_alpha,num_M*num_Obs))
    # Integrating out the information/significance levels with agression
    Scores = T.dot(Agression.T,Scores)
    # Finally get the num_M by num_Obs matrix of scores for each observation
    Scores = T.reshape(Scores,(num_Obs,num_M)).T
    # Return the model choice
    Choice = T.argmax(Scores, axis = 0)
    
    return Choice

def get_pvalues(M, Obs, K, num_M, num_Obs, pValue_alg = 0):
    t0 = time.time()
    if pValue_alg == 0:
        pvalues = chi2_pvalues(M, Obs, K, num_M, num_Obs)    
    elif pValue_alg == 1:
        models = M.get_value()
        obs = Obs.get_value()
        pvalues = direct_pvalues(models, obs, K, num_M, num_Obs)
    t1 = time.time()

    print 'pvalues took ', (t1-t0)/60.,' min'
    
    return pvalues    

def call_underfit_choice_theano(M, Obs, num_M, num_Obs, K, num_alpha, agression_profiles = None, Preds = None, pValue_alg = 0):
    #theano.config.compute_test_value = 'warn'
    
    if agression_profiles == None:
        agression_profiles = np.ones((num_alpha, 1)/num_alpha, dtype = theano.config.floatX)
    
    pvalues = get_pvalues(M, Obs, K, num_M, num_Obs, pValue_alg = pValue_alg)
    
    #Alpha = T.matrix # 1 x num_alpha
    
    alpha = theano.shared(np.asmatrix(np.linspace(0.0,1.0, num = num_alpha, endpoint = False), dtype=theano.config.floatX))
    Agression_profiles = T.matrix('Agr')
    nAlpha, nM, nO = T.iscalars('','','')
    
    # scan over the agression profiles
    agression_choices, _ = theano.scan(fn = underfit_choice, outputs_info = None, sequences = Agression_profiles , non_sequences = [nAlpha, M, Obs, K, nM, nO, pvalues, alpha, Preds])

    # returns a list of  lists of model choices per observation for each agression function
    f = theano.function([Agression_profiles, nAlpha, nM, nO], agression_choices, allow_input_downcast = True)
    
    return f(agression_profiles, num_alpha, num_M, num_Obs)


'''
Demonstrating functionality:
''
#from generate_models import *
#from chi2pvalue import *
import cPickle as pickle
f=open('../models.pkl','rb')
l = pickle.load(f)[0]
f.close()

K = 2
#l = generate_models(K,0.1)
num_M = len(l)

M = theano.shared(np.asarray(l))
Obs = theano.shared(np.asarray([[1,2],[3,2],[0,2]]))
num_Obs = 3

num_alpha = 10
Agression = theano.shared(np.ones((num_alpha,1), dtype = float)/num_alpha)

pvalues = direct_pvalues(l, Obs, K, num_M, num_Obs)

report = underfit_choice(Agression, num_alpha, M, Obs, K, num_M, num_Obs, pvalues)
f = theano.function([],report)
print f()

from choice_to_models import choice_to_models
print choice_to_models(f(),l,num_Obs)

#'''
