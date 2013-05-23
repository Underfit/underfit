import numpy as np
from scipy.stats import moment


def get_predictiveness_array(choices, verses, Preds, num_Obs):

    array = np.ones((num_Obs))
    for i in xrange(num_Obs):
        array[i] = Preds[int(verses[i])][int(choices[i])]
    return array
    
def get_moments(pred_array, num_Obs):
    first = np.mean(pred_array)
    second = moment(pred_array, moment = 2)
    third = moment(pred_array, moment = 3)
    
    return [first, second, third]
