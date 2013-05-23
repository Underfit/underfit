import numpy as np
from generate_models import generate_models
import cPickle 

models = []
for i in xrange(8):
    models.append(np.asmatrix(generate_models(i+2)))

    
f = open("models.pkl", "wb")    
cPickle.dump(models,f)
f.close()