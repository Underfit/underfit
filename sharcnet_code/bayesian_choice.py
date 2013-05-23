import theano 
import numpy as np
from theano import tensor as T
try:
    from theano.sandbox.linalg.kron import kron
except:
    from theano.toolbox import kron
from theano.ifelse import ifelse    

from multinomial_probabilities import multinomial_probabilities
from predictiveness import predictiveness_profiles

#The Choice_type is 1,2,3, or 4, corresponding to 4 different Bayesian choice rules

# The Bayesian choice is a function of the posterior and a loss function (here a function of the predictiveness profiles)
# 1 - Minimize the expectation of (1 - predictiveness(m;X))^2, with X distributed by the posterior
# 2 - Minimizes the expectation of |1 - predictiveness(m;X)|
# 3 - Maximizes the expectation of predictiveness(m;X)
# 4 - Chooses the model with maximum a posteriori probability
#- maximum a posteriori: highest posterior probability gets the choice
#Models is num_M by K
#Priors is num_M by 1
#Obs is num_Obs by K
def bayesian_choice(Choice_type, Priors, Models, Obs, K, num_M, num_Obs):
    #Calculating the posterior distribution; it is proportional to Prior(M) * Likelihood(M|D)
    #Posterior is num_M by num_Obs, and holds a probability for each model for each trial
    Posterior = (multinomial_probabilities(Models, Obs, K, num_M, num_Obs).T*Priors).T
    normalizer = T.sum(Posterior, axis = 0)
    Posterior = Posterior/normalizer
    
    def min_risk_choice(Posterior):

        #The Loss function is a function of the predictiveness profiles
        Preds = predictiveness_profiles(Models, K, num_M)
        
        Loss = ifelse(T.eq(Choice_type, 1), T.pow(1.0 - Preds,2), ifelse(T.eq(Choice_type, 2), T.abs_(1.0 - Preds), - Preds))             
        
        #Kroneckering Loss up num_Obs times (tile Loss, making it num_M by num_M*num_Obs)
        Loss = kron(T.ones((1,num_Obs)), Loss)        
        #Kroneckering up the Posterior, making it num_M by num_Obs*numM
        Posterior = kron(Posterior, T.ones((1,num_M)))

        #Dotting and reshaping down to give num_M by num_Obs expected loss matrix
        Expected_Loss = T.dot(T.ones((1,num_M)),Posterior*Loss)            
        Expected_Loss = T.reshape(Expected_Loss, (num_Obs,num_M)).T
        
        #Choice minimizes risk
        Choice = T.argmin(Expected_Loss, axis = 0) 
        return Choice 

    #Fourth choice (which is MLE if the prior is uniform) doesn't require a loss function:
    Choice = ifelse(T.eq(Choice_type, 4), T.argmax(Posterior, axis=0), min_risk_choice(Posterior))    

    return Choice

def call_bayesian_choice_theano(M, Obs, num_M, num_Obs, K, priors_profiles, loss_function = 1):
    
    Priors_profiles = T.matrix('Priors')
    Loss_funcs = T.arange(1,5)
    nM, nO = T.iscalars('','')
    
    def scan_over_loss_functions(priors, M, Obs, nM, nO, K, Loss_funcs):
        # scan over loss functions
        choices_for_loss , _ = theano.scan(bayesian_choice, outputs_info = None,  sequences = Loss_funcs, non_sequences=[priors, M, Obs, K, nM, nO])

        return choices_for_loss 
    
    # scan over priors 
    choices_profile, _ = theano.scan(scan_over_loss_functions,  outputs_info = None, sequences = Priors_profiles, non_sequences =[M, Obs, nM, nO, K, Loss_funcs])
    
    f = theano.function([Priors_profiles, nM, nO], choices_profile)
    
    return f(priors_profiles, num_M, num_Obs)
   
     
'''
Demonstrating functionality:
''
K = 2
import numpy as np
from generate_models import *
l = generate_models(K,0.1)
num_M = len(l)
M = theano.shared(np.asarray(l))
Obs = theano.shared(np.asarray([[1,2],[3,2],[0,2]]))
num_Obs = 3

#priors
P = T.ones((num_M,1))
for i in xrange(1,5):
    probs = bayesian_choice(i, P, M, Obs, K, num_M, num_Obs)
    f = theano.function([],probs)
    print np.round(f(), 2)
#'''