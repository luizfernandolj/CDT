from absc.detector import Detector
from utils.generate_samples import generate_samples_binary
from quantification.dys_method import get_dys_distance
from utils.get_train_values import get_train_values
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from moss import MoSS


class CDT(Detector):
    
    def __init__(self,classifier, train_split_size:float=0.5, n_train_test_samples:int=100, p:int=1) -> None:
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
    
    def fit(self, ref_window: pd.DataFrame) -> None: 
        self.ref_window = ref_window
        
        X = ref_window.drop(["class", "context"], axis=1)
        Y = ref_window["class"]
        
        self.pos_scores, self.neg_scores, self.classifier = get_train_values(X, Y, 10, self.classifier)
        
        
        for m in np.arange(0.1, 1, 0.1):
            scores = MoSS(len(ref_window), m, self.pos_scores, self.neg_scores)
            sns.histplot(scores)
            plt.show() 
            dys_distance = get_dys_distance(self.pos_scores, self.neg_scores, scores)
            print(dys_distance)
            self.distances.append(dys_distance)    
            
        
        self.threshold = np.percentile(self.distances, [self.p, (100-self.p)])

    
        
    def detect(self, current_window: pd.DataFrame) -> bool:
        test_scores = self.classifier.predict_proba(current_window)
        dys_distance = np.round(get_dys_distance(self.pos_scores, self.neg_scores, test_scores), 5)
        print(dys_distance, self.threshold)
        if dys_distance < self.threshold[0] or dys_distance > self.threshold[1]:
            return True
        return False