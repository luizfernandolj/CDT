import pandas as pd
import numpy as np
from DataStream.base import Detector

class BASELINE(Detector):
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        pass
        
    
    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame) -> None:
        pass
    
    def detect(self, current_window:pd.DataFrame=None) -> bool:
        return False