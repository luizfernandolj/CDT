from absc.detector import Detector
from utils.generate_samples import generate_samples_binary
from quantification.dys_method import get_dys_distance
from utils.get_train_values import get_train_values
import pandas as pd
import numpy as np


class CDT(Detector):
    
    def __init__(self, classifier: np.any ,train_split_size:float=0.5, n_samples:int=100) -> None:
        self.train_split_size = train_split_size
        self.train = None
        self.distances = []
        self.current_window = None
        self.threshold = None
        self.ref_window = None
        self.n_samples = n_samples
        self.classifier = classifier
        
        
    def __call__(self, current_window: pd.DataFrame) -> None:
        self.current_window = current_window
             
        
        
    def fit(self, ref_window: pd.DataFrame) -> None: 
        size = int(len(ref_window) * self.train_split_size)
        self.ref_window = ref_window
        
        self.train = ref_window.iloc[:size].reset_index(drop=True)
        X = self.train.drop("class", axis=1)
        Y = self.train["class"]
        
        
        pos_scores, neg_scores, self.classifier = get_train_values(X, Y, 10, self.classifier)
        
        test = ref_window.iloc[size:].reset_index(drop=True)
        
        samples = generate_samples_binary(test, self.n_samples, int(len(self.train)/1.5))
        
        for sample in samples:
            test_scores = self.classifier.predict_proba(sample)
            dys_distance = np.round(get_dys_distance(pos_scores, neg_scores, test_scores), 3)
            print(dys_distance)
            self.distances.append(dys_distance)

    
        
    def detect(self) -> bool:
        # TODO
        return False