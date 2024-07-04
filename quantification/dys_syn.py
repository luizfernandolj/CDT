
import numpy as np
import utils.DyS_utils as qntu
import time
from moss import MoSS
    
def get_dys_distance(test_scores, alpha_train=0.5, n=1000, measure='topsoe'):
    bin_size = np.linspace(2,20,10)   #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    
    distance = {}
    for m in np.linspace(0.1, 0.4, 4):
        pos_scores, neg_scores = MoSS(n, alpha_train, m)
        result  = []
        for bins in bin_size:
            #....Creating Histograms bins score\counts for validation and test set...............
            
            p_bin_count = qntu.getHist(pos_scores, bins)
            n_bin_count = qntu.getHist(neg_scores, bins)
            te_bin_count = qntu.getHist(test_scores, bins) 
            
            def f(x):            
                return(qntu.DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
        
            result.append(qntu.TernarySearch(0, 1, f))                                           
                        
        pos_prop = np.median(result)
        
        index = np.where(sorted(result) == pos_prop)[0].astype(int)[0]
        p_bin_count = qntu.getHist(pos_scores, bin_size[index])
        n_bin_count = qntu.getHist(neg_scores, bin_size[index])
        te_bin_count = qntu.getHist(test_scores, bin_size[index])
        
        distance[m] = qntu.DyS_distance(((p_bin_count*pos_prop) + (n_bin_count*(1-pos_prop))), te_bin_count, measure = measure)
    
    return distance

    
        




