import numpy as np

def divisible_dataset(size, num_groups, upper_bound_x=5000, normalize=True):
    x = np.random.randint(low=1, high=5000, size=size)
    y = np.zeros(size)

    for i in range(size):
      for j in range(1, num_groups+1):
        if x[i] % j == 0:
          y[i] += np.random.randn()
    
    if normalize:
        y = np.array([abs(label)/(1+abs(label)) for label in y])

    g = lambda x: [(x % j == 0) for j in range(1, num_groups+1)]
    
    return x, y, g
