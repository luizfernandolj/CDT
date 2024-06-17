import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def MoSS(n, m, pos, neg):

  # Calculate the first shifted sigmoid function (weight: alpha)
  syn_plus = np.power(pos, m)  # Exponent `m` affects difficulty

  # Calculate the second shifted sigmoid function (weight: 1-alpha)
  syn_minus = 1 - np.power(1 - neg, m)  # Exponent `m` affects difficulty

  # Combine both functions using set union
  moss = np.union1d(syn_plus, syn_minus)
  print(moss)

  return moss

