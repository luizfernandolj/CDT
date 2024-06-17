from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class Detector(ABC):
    """Abstract class for drift detectors"""
    
    @abstractmethod
    def fit(self, ref_window:pd.DataFrame) -> None:
        pass
    
    @abstractmethod
    def detect(self, current_window:pd.DataFrame) -> bool:
        pass
    
    