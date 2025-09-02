import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 256
h = (2*np.pi)/n
x_grid = np.arange(h, 2*np.pi + h, h)
y_grid = np.arange(h, 2*np.pi + h, h)
X, Y = np.meshgrid(x_grid, y_grid)

# Read binary data
with open('vor2500.dat', 'rb') as f:
    dum1 = np.fromfile(f, dtype=np.float32, count=1)        # first dummy value
    omega = np.fromfile(f, dtype=np.float64, count=n*n)     # main data
    dum2 = np.fromfile(f, dtype=np.float32, count=n)        # second dummy block

# Reshape and transpose
omega = omega.reshape((n, n))

