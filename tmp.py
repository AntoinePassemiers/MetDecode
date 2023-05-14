import numpy as np
from matplotlib import pyplot as plt

data = np.load('tmp.npz')

plt.scatter(data['R_atlas'][0], data['R_corrected'][0], alpha=0.4)
plt.show()
