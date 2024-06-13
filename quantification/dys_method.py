
import numpy as np
import utils.DyS_utils as qntu
import time

def dys_method(pos_scores, neg_scores, test_scores, measure='topsoe'):
    """DYs Framework

    It quantifies events based on framework of different distance measures that caculate the distance between the mixture of positive and negative scores and the test scores, applying DYs framework, according to Maletzke et al.(2019).
    
    Parameters
    ----------
    Positive scores : array
        A numeric vector of positive scores estimated from the validation set using 10-fold stratified cross-validation.
    Negative scores : array
        A numeric vector of Negative scores estimated from the validation set using 10-fold stratified cross-validation.
    Test scores : array
        A numeric vector of scores predicted from the test set.
    measure : string
        The distance metric used to claculate the distance between mixture and test distributions. the default measure is 'topose' 
        
    Returns
    -------
    array
        the class distribution of the test. 
    """
    
    bin_size = np.linspace(2,20,10)   #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    
    result  = []
    start = time.time()
    for bins in bin_size:
        #....Creating Histograms bins score\counts for validation and test set...............
        
        p_bin_count = qntu.getHist(pos_scores, bins)
        n_bin_count = qntu.getHist(neg_scores, bins)
        te_bin_count = qntu.getHist(test_scores, bins)
        
        def f(x):            
            return(qntu.DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
    
        result.append(qntu.TernarySearch(0, 1, f))                                           
                        
    pos_prop = np.median(result)
    stop = time.time()
    #return stop - start

    return pos_prop
    
def get_dys_distance(pos_scores, neg_scores, test_scores, measure='topsoe'):
    bin_size = np.linspace(2,20,10)   #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    
    result  = []
    start = time.time()
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
    
    distance = qntu.DyS_distance(((p_bin_count*pos_prop) + (n_bin_count*(1-pos_prop))), te_bin_count, measure = measure)
    
    return distance

    
        




