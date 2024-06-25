from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class Detector(ABC):
    """Abstract class for drift detectors"""
    
    @abstractmethod
    def fit(self, X_ref_window:pd.DataFrame, y_ref_window:pd.DataFrame) -> None:
        pass
    
    @abstractmethod
    def detect(self, current_window:pd.DataFrame) -> bool:
        pass
    
    