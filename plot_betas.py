import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import keras
from model import build_model
from sampler import RandomWalkSampler
from datasets import seqs_to_array
from itertools import product
import cPickle


def generate_transition_matrix(k):
    transition_matrix = np.zeros((k**2, k**2))
    span = xrange(k)
    disp = [-1, 0, 1]
    for i, j in product(span, span):
        pos = i*k + j
        for di, dj in product(disp, disp):
            x = i + di
            y = j + dj
            neighb_pos = x*k + y
            test1 = (di != 0) or (dj != 0)
            test2 = (0 <= x < k)
            test3 = (0 <= y < k)
            if test1 and test2 and test3:
                transition_matrix[pos][neighb_pos] = 1
    return transition_matrix


def create_samples(dataset_size, k):
    from datasets import seqs_to_array
    out = seqs_to_array([rw_sampler.gen_sequence(25) for _ in
                         xrange(dataset_size)], vocab = range(k**2))
    return out


with open('./data/sampler.pkl', 'rb') as pkl:
    rw_sampler = cPickle.load(pkl)

train = create_samples(1000, 10)
transition_matrix = generate_transition_matrix(10)
model = build_model(shape=(train.shape[1] - 1, train.shape[2]),
                    transition_matrix=transition_matrix,
                    accumulator_method='binary',
                    z_dim=None)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x=train[:,:-1,:],
          y=train[:,1:,:],
          batch_size=100,
          epochs=300)

true_betas = rw_sampler.beta
learned_betas = model.get_layer('time_distributed_1').get_weights()[0]
learned_betas = np.diag(learned_betas).reshape(10, 10)

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,2,1)
ax.imshow(np.log(true_betas), cmap=plt.cm.PuOr, vmin=-np.log(10)*3,
          vmax=np.log(10)*3)
ax.tick_params(length=0, labelbottom='off', labelleft='off')

ax = fig.add_subplot(1,2,2)
ax.imshow(learned_betas, cmap=plt.cm.PuOr, vmin=-np.log(10)*3,
          vmax=np.log(10)*3)
ax.tick_params(length=0, labelbottom='off', labelleft='off')

plt.tight_layout()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.savefig('betas.png')
plt.show()
# def plot_states(sampler):
#     sampler.reset()
#     generator = sampler.gen_sample()
#     pos = sampler.pos
#     fig = plt.figure(figsize=(9, 4))
#     for i in xrange(1, 11):
#         ax = fig.add_subplot(2, 5, i)
#         ax.grid('on', color='black', linestyle='-', linewidth=1)
#         ax.imshow(-np.log(sampler.beta ** sampler.x),
#                    cmap=plt.cm.PuOr,
#                    vmin=-np.log(10)*3,
#                    vmax=np.log(10)*3)
#         ax.scatter(pos[-1], pos[0], marker='X', color='black')
#         ax.tick_params(length=0, labelbottom='off', labelleft='off')
#         ax.set_xticks(np.arange(-.5,5,1))
#         ax.set_yticks(np.arange(-.5,5,1))
#         pos = next(generator)
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     plt.tight_layout()
#     plt.savefig('simulation_demo.png')
# 
# 
# if __name__ == '__main__':
#     rw_sampler = RandomWalkSampler(k=5, betas=[1/10., 10.],
#                                    homeward_bound=False, readable_states=True)
#     plot_states(rw_sampler)
# 
