from absc.detector import Detector
from utils.generate_samples import generate_samples_binary
from quantification.dys_method import get_dys_distance
from utils.get_train_values import get_train_values
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from moss import MoSS, calculate_merging_factor


class CDT(Detector):
    
    def __init__(self,classifier, train_split_size:float=0.5, n_train_test_samples:int=100, p:int=10) -> None:
        self.train_split_size = train_split_size
        self.distances = []
        self.threshold = None
        self.p = p
        self.ref_window = None
        self.classifier = classifier
        self.n_train_test_samples = n_train_test_samples
        self.pos_scores = None
        self.neg_scores = None
        
        
    # window
    # separate between train and test
    # select from train and test where context equals the new context
    # get scores from train
    # create multiple tests
    # get dys distance from each test set
    def __call__(self, current_window: pd.DataFrame) -> None:
        pass
    
    
    def fit(self, ref_window: pd.DataFrame) -> None: 
        self.ref_window = ref_window.drop("context", axis=1)
        
        X = self.ref_window.drop(["class"], axis=1)
        Y = self.ref_window["class"]
        
        self.pos_scores, self.neg_scores, self.classifier = get_train_values(X, Y, 10, self.classifier)
        
        m = calculate_merging_factor(self.ref_window.iloc[-(int(len(ref_window)/2)):])
        
        for alpha in np.arange(0, 1, 0.1):
            scores = MoSS(len(ref_window), alpha, m)
            dys_distance = get_dys_distance(self.pos_scores, self.neg_scores, scores)
            self.distances.append(dys_distance)  
            
        
        self.threshold = np.percentile(self.distances, [self.p, (100-self.p)])
        print(self.threshold)
    
        
    def detect(self, current_window: pd.DataFrame) -> bool:
        test_scores = self.classifier.predict_proba(current_window)
        dys_distance = np.round(get_dys_distance(self.pos_scores, self.neg_scores, test_scores), 5)
        if dys_distance < self.threshold[0] or dys_distance > self.threshold[1]:
            print(dys_distance)
            return True
        return False