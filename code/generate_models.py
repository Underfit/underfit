
#This python script generates the set of probability vectors with each entry evenly divisible by step_width
def generate_models(K, step_width = 0.1): 
    pmf_store = []
    probs_list = []
    recursive_give_all_pmfs(K, step_width, pmf_store, probs_list)
    return pmf_store

#It's recursive
from copy import deepcopy
def recursive_give_all_pmfs(num_terms_remaining, step_width, pmf_store, probs_list, space_left = 1.0):

    if (num_terms_remaining == 1):
        probs_list.append(abs(round(space_left,3)))
        pmf_store.append(deepcopy(probs_list))
        probs_list.pop()
    else:
        spots_left = int(round((space_left/step_width + 1),3))
        for i in xrange(spots_left):
            p = round(i*step_width,3)
            probs_list.append(p)
            space_left -= p
            num_terms_remaining = num_terms_remaining - 1
                        
            recursive_give_all_pmfs(num_terms_remaining, step_width, pmf_store, probs_list, space_left)
            
            probs_list.pop()
            space_left = space_left + p
            num_terms_remaining = num_terms_remaining + 1 
