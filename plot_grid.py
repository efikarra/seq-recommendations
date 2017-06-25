import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sampler import RandomWalkSampler


def plot_states(sampler):
    sampler.reset()
    generator = sampler.gen_sample()
    pos = sampler.pos
    fig = plt.figure(figsize=(9, 4))
    for i in xrange(1, 11):
        ax = fig.add_subplot(2, 5, i)
        ax.grid('on', color='black', linestyle='-', linewidth=1)
        ax.imshow(-np.log(sampler.beta ** sampler.x),
                   cmap=plt.cm.PuOr,
                   vmin=-np.log(10)*3, vmax=np.log(10)*3)
        ax.scatter(pos[-1], pos[0], marker='X', color='black')
        ax.tick_params(length=0, labelbottom='off', labelleft='off')
        ax.set_xticks(np.arange(-.5,5,1))
        ax.set_yticks(np.arange(-.5,5,1))
        pos = next(generator)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout()
    plt.savefig('simulation_demo.png')


if __name__ == '__main__':
    rw_sampler = RandomWalkSampler(k=5, betas=[1/10., 10.],
                                   homeward_bound=False, readable_states=True)
    plot_states(rw_sampler)

