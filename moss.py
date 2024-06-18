import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def MoSS(n:int, alpha:float, m:float):
  
  n_pos = int(n*alpha)
  n_neg = int((1-alpha)*n)

  x_pos = np.arange(1, n_pos, 1)
  x_neg = np.arange(1, n_neg, 1)
  
  syn_plus = np.power(x_pos/(n_pos+1), m)
  syn_neg = 1 - np.power(x_neg/(n_neg+1), m)

  moss = np.union1d(syn_plus, syn_neg)

  return moss



def calculate_merging_factor(data):

  # Calculate the proportion of the positive class
  positive_count = len(data[data.iloc[:, -1] == 1])
  # Calculate the proportion of the positive class
  p = positive_count / len(data) 

  # Calculate the Gini coefficient
  gini = 2 * p * (1 - p)

  # Calculate the merging factor
  m = np.round(1 - np.sqrt(gini), 1)

  return m
