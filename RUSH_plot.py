import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict


def load_data(fname):
    out = defaultdict(list)
    with open(fname, 'r') as f:
        for line in f:
            name_size, data = line.split('\t')
            data = float(data)
            name, size = name_size.split()
            out[name].append((size, data))
    return out


if __name__ == '__main__':
    loss_data = load_data('./data/saved-models/RUSH_valid.txt')
    time_data = load_data('./data/saved-models/RUSH_time.txt')

    params = {
       'axes.labelsize': 10,
       'font.size': 10,
       'legend.fontsize': 10,
       'xtick.labelsize': 10,
       'ytick.labelsize': 10,
       'text.usetex': False
    }
    mpl.rcParams.update(params)
    cmap = plt.cm.get_cmap('tab10')

    fig = plt.figure(figsize=(9, 4))

    ax1 = fig.add_subplot(121)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(axis='y', length=0)
    ax1.set_xlabel('Dataset size')
    ax1.set_xscale('log')
    ax1.set_ylabel('Test loss')
    ax1.grid(color='#dddddd', linewidth=0.5)
    ax1.xaxis.grid(False)
    for i, item in enumerate(loss_data.iteritems()):
        name, value = item
        x, y = zip(*value)
        ax1.plot(x, y, label=name, color=cmap(i))
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', length=0)
    ax2.set_xlabel('Dataset size')
    ax2.set_xscale('log')
    ax2.set_ylabel('Train time (hours)')
    ax2.grid(color='#dddddd', linewidth=0.5)
    ax2.xaxis.grid(False)
    for i, item in enumerate(time_data.iteritems()):
        name, value = item
        x, y = zip(*value)
        y = [time / 3600. for time in y] # to hours
        ax2.plot(x, y, label=name, color=cmap(i))

    plt.tight_layout()
    plt.savefig('simulation.png')

