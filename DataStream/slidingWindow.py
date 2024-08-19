import pandas as pd
import numpy as np

class SlidingWindow:
    """SlidingWindow class for creating a sliding window over a data stream with features, labels, and optional context."""
    
    def __init__(self, X, y, context=None, window_size:int=5) -> None:
        self.window_size = window_size
        self.index = 0
        
        # Create the initial window (reference window) and stream
        self.starting_window = self._create_window(X, y, context, start=0, size=window_size)
        self.stream = self._create_stream(X, y, context)
        self.actual_window = self.starting_window.copy()
    
    def _create_window(self, X, y, context, start, size):
        """Creates a window of data given the starting point and size."""
        end = start + size
        if context is not None:
            return pd.concat([X.iloc[start:end], y.iloc[start:end], context.iloc[start:end]], axis=1).reset_index(drop=True)
        return pd.concat([X.iloc[start:end], y.iloc[start:end]], axis=1).reset_index(drop=True)
    
    def _create_stream(self, X, y, context):
        """Creates the complete data stream."""
        if context is not None:
            return pd.concat([X, y, context], axis=1).reset_index(drop=True)
        return pd.concat([X, y], axis=1).reset_index(drop=True)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.stream) - self.window_size + 1:
            raise StopIteration
        
        # Slide the window
        self.actual_window = self.stream.iloc[self.index:self.index + self.window_size, :].copy().reset_index(drop=True)
        self.index += 1
        
        X, y = self.actual_window.iloc[:, :-2], self.actual_window.iloc[:, -2]
        context = self.actual_window.iloc[:, -1] if self.actual_window.shape[1] == X.shape[1] + 2 else None
        return Window(X, y, context)
    
    def __call__(self, func, *args):
        return func(Window(self.starting_window.iloc[:, :-2], self.starting_window.iloc[:, -2], 
                           self.starting_window.iloc[:, -1] if self.starting_window.shape[1] == X.shape[1] + 2 else None),
                    Window(self.actual_window.iloc[:, :-2], self.actual_window.iloc[:, -2], 
                           self.actual_window.iloc[:, -1] if self.actual_window.shape[1] == X.shape[1] + 2 else None), 
                    *args)
    
    def get_actual_instance(self, with_context=False):
        if with_context and self.actual_window.shape[1] == self.actual_window.iloc[:, :-2].shape[1] + 2:
            return self.actual_window.iloc[-1, :].reset_index(drop=True)
        return self.actual_window.iloc[-1, :-1].reset_index(drop=True)
    
    def get_actual_context(self):
        if self.actual_window.shape[1] == self.actual_window.iloc[:, :-2].shape[1] + 2:
            return self.actual_window.iloc[-1, -1]
        raise KeyError("Context not available.")
    
    def set_context(self, context_list: list):
        self.stream["context"] = context_list
    
    def switch(self):
        self.starting_window = self.actual_window.copy()
