import matplotlib.pyplot as plt
import numpy as np
from corner import corner

data = np.load("chain.npz")

x = data["x"]
a = data["a"]

corner(x.T)
plt.show()

plt.plot(x[0, :])
plt.show()
