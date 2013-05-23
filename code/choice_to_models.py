import numpy as np
def choice_to_models(choice, models, num_Obs):
    l = []
    for i in xrange(num_Obs):
        l.append(models[choice[i]])
    return np.asmatrix(l)