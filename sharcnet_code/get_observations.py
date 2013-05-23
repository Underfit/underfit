import numpy as np
from numpy import random

def get_obs(model, sample_size):
    log = []
    for i in xrange(sample_size):
        rv = random.uniform(0.0,1.0)
        for h in xrange(len(model)):
            rv -= model[h]
            if rv < 0:
                log.append(h)                
                break
    obs = []    
    for i in xrange(len(model)):
        obs.append(0)
    
    for o in log:
        obs[o] += 1
    return obs
