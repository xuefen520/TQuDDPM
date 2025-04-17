#%%
# plot loss decay
import jax.numpy as jnp
import numpy as np

from src.distance_jax import naturalDistance
from src.QDDPM_jax import HaarSampleGeneration, QDDPM, setDiffusionDataMultiQubit
import matplotlib.pyplot as plt

n = 4
N = 10000
T = 20
loss_T = []

X = jnp.load('tfimDiff_n4T30_N10000.npy')
X_np = np.array(X)
np.save('tfimDiff_n4T30_N10000_np.npy', X_np)
XTref = HaarSampleGeneration(N, 2**n, seed=40)
for t in range(T+1):
    loss_T.append(naturalDistance(X[t], XTref))
print(loss_T)
# np.save('data/phase/tfimDiffdis_n4T20_N10000.npy', loss_T)
# print(X)
plt.plot(loss_T)
plt.show()