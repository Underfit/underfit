import numpy as np
import theano
import theano.tensor as T

A = T.matrix()

out = theano.function([A], A**2)

a = np.random.randn(3, 3)
print a

print out(a)
