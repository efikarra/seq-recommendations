import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

x = [10, 100, 1000, 10000]
count = [1.97, 1.31, 0.92, 0.93]
z2 = [2.1, 2.01, 1.64, 1.52]
z10 = [2.58, 2.16, 1.23, 0.97]

fig = file('data/saved-models/dataset-effect.png', 'wb')
ax = plt.subplot(111)
ax.set_title('Effect of dataset size')
ax.set_xlabel('Dataset size')
ax.set_ylabel('Validation loss')
ax.plot(x, count, label='x')
ax.plot(x, z2, label='z=2')
ax.plot(x, z10, label='z=10')
ax.legend()
plt.savefig(fig)
fig.close()

