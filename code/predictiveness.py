import theano 
from theano import tensor as T
import numpy as np

#Global variable for making log(0) = -INF
INF = 1000000

# This function returns the predictiveness of each model (row of M) against each other (row of M)
# Returns profiles as columns of an num_M by num_M tensor 
# Predictiveness(m;q) = exp(-D_kl(q;m))
def predictiveness_profiles(M, K, num_M): 
    #Fixing the zeros
    #fM = theano.function([],M)
    #mZeros = theano.shared(np.asarray(fM()==0, dtype = theano.config.floatX))
    
    mZeros = T.eq(M, 0)
    
    logM = (T.log(M + mZeros) - INF*mZeros)
    
    #Calculating the entropy terms; sum  p_i ln p_i    
    H = -T.dot(M*logM, T.ones((K,1)))

    #And the cross entropy terms; sum  p_i ln m_i
    U = T.dot(M, logM.T)

    #Finally the predictiveness        
    P = T.exp(H)*T.exp(U)
    
    return P
 
'''
Demonstrating functionality:
''

from generate_models import *
l = generate_models(2,0.1)
num_M = len(l)
print np.asarray(l)
M = theano.shared(np.asarray(l))
fM = theano.function([],predictiveness_profiles(M, 2, 11))
print np.round(fM(),2)
#'''
