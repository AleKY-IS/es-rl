import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


x = np.linspace(0, 31, 200)
y = np.sin(x)

noise = np.random.randn(200)

y_noisy = y + noise

plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, '-')
ax.plot(x, y_noisy, 'x')

embed()