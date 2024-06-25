import pandas as pd
import numpy as np
from absc.detector import Detector
from detectors.iks_code.IKSSW import IKSSW

class IKS(Detector):
    
    def __init__(self, ca:float) -> None:
        self.ca = ca
        self.ikssw = None
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        instance = current_window.iloc[-1].values.tolist()
        
        self.ikssw.Increment(instance)
        
    
    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame) -> None:
        self.ikssw = IKSSW(X_ref_window.values.tolist())
    
    def detect(self, current_window:pd.DataFrame=None) -> bool:
        return self.ikssw.Test(self.ca)