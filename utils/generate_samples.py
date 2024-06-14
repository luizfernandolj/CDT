import pandas as pd
import numpy as np
from typing import Generator

def generate_samples_binary(batch:pd.DataFrame, 
                            n_samples:int=100,
                            sample_size:int=100, 
                            label_column:str="class") -> Generator:
    
    alphas = np.random.rand(1, n_samples)
    sample_size = len(batch/2) if sample_size >= len(batch) else sample_size
    
    positives = batch.loc[batch[label_column] == 1]
    negatives = batch.loc[batch[label_column] == 0]
    
    for alpha in alphas[0]:
        pos_class_size = int(sample_size * alpha)
        neg_class_size = sample_size - pos_class_size
        
        if pos_class_size is not sample_size:
            positive_samples = positives.sample(pos_class_size, replace=True)
        else:
            positive_samples = negatives.sample(frac=1, replace=True)

        negative_samples = negatives.sample(int(neg_class_size), replace=True)
        
        test_sample = pd.concat([positive_samples, negative_samples])
        
        yield test_sample.drop(label_column, axis=1)
    
    