import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,6))
ax = plt.subplot(111)
with open('./data/saved-models/beta-negative/dev_log.txt', 'r') as f:
    ax.set_title('Negative Beta')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    for i, line in enumerate(f):
        id, data = line.split('\t')
        data = [float(x) for x in data.split()]
        ax.plot(data, label=id, color=plt.cm.get_cmap('Set2')(i))
    ax.legend()
plt.savefig('beta-negative.png')
