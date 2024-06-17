import pandas as pd
import numpy as np
from absc.detector import Detector
from iks_code.IKSSW import IKSSW

class IKS(Detector):
    
    def __init__(self, ca:float) -> None:
        self.ca = ca
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        instance = current_window.iloc[-1].values.tolist()
        
    
    def fit(self, ref_window:pd.DataFrame) -> None:
        pass
    
    def detect(self, current_window:pd.DataFrame) -> bool:
        pass