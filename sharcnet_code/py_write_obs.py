#from write_observations import write_observations as write_py
from Cwrite_observations import write_observations as write_c

import time

t0 = time.time()
write_c()
t1 = time.time()
print t1-t0