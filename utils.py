import numpy as np
import matplotlib.pyplot as plt
epsilon=10e-8


def plot(values, colors, labels, ylabel, xlabel, save_path="test_fig"):
    for i,vals in enumerate(values):
        plt.plot(vals, color=colors[i], label=labels[i])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.close()

def rank_plot(seqs, vocab, title="flickr data", save_path=None):
    alpha=multinomial_probabilities(seqs, len(vocab), k=1.0, normalize=False)
    alpha.sort(axis=1)
    alpha = alpha[:,::-1]
    plt.xlabel("state")
    plt.ylabel("frequency")
    plt.title(title)
    plt.bar(range(len(vocab)),alpha[0,:])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def multinomial_probabilities(seqs, n, k=1.0, normalize=True):
    """Learn multinomial probabilities from sequences

        Args:
            seqs: Contains sequences to learn from.
            n: number of states.

        Returns:
            alpha: Probabilities vector in the form of an np.array.
        """
    alpha = np.zeros((1, n))
    # Fill with counts
    for seq in seqs:
        for s in seq:
            alpha[0, s] = alpha[0, s] + 1
    # Normalize
    z = np.sum(alpha, axis=1)
    alpha = (alpha + k)
    if normalize:
        alpha = alpha / (z + n * k)
    return alpha


def transition_matrix(seqs, n, k=0, freq=False, end_state=True):
    """Learn global Markov transition matrix from sequences

    Args:
        seqs: Contains sequences to learn from.
        vocab: Words in sequences.
        k: Smoothing parameter from Dirchlet prior.
        prob: If True then matrix returned is transition probabilities,
            otherwise transition counts are returned.
        end_state: If True then adds a token for the end state.

    Returns:
        T: Transition matrix in the form of an np.array.
    """
    if end_state:
        alpha = np.zeros((n, n + 1))  # Note: +1 for end_token
    else:
        alpha = np.zeros((n, n))
    gamma = np.zeros(n)
    # Fill with counts
    for seq in seqs:
        if len(seq) > 1:
            for i, j in zip(seq[:-1], seq[1:]):
                alpha[i, j] = alpha[i, j] + 1
            if end_state:
                alpha[j, n] = alpha[j, n] + 1
        else:
            if end_state:
                alpha[seq[0], n] = alpha[seq[0], n] + 1
        gamma[seq[0]] = gamma[seq[0]] + 1
    smoothed_alpha = (alpha + k)
    smoothed_gamma= (gamma + k)
    if not freq:
        z = np.sum(alpha, axis=1).reshape((n, 1))
        smoothed_alpha /= (z + n * k)
        smoothed_gamma /= (np.sum(gamma)+n * k)
    return smoothed_alpha, smoothed_gamma



def neg_log_likelihood(probs):
    return -np.sum(np.log(probs))


def compute_likelihood_cut(predictions, train_percent, orig_lengths=None, count_first_prob=False):
    assert train_percent <= 1.0, "ERROR: train_percent should be <= 1.0"
    train_lls=[]
    val_lls=[]
    for i,pred in enumerate(predictions):
        sort_pred=pred[:]
        if not count_first_prob:
            sort_pred=sort_pred[1:]
        if orig_lengths is not None:
            sort_pred=pred[-int(orig_lengths[i]):]
        seq_length = len(sort_pred)
        train_elems = int(np.ceil(train_percent * seq_length))
        val_elems = int(np.floor((1.0-train_percent) * seq_length))
        if train_elems > 0:
            train_lls.append(neg_log_likelihood(sort_pred[0:train_elems])/train_elems)
        if val_elems>0 :
            val_lls.append(neg_log_likelihood(sort_pred[-val_elems:])/val_elems)

    return np.sum(train_lls)/len(train_lls),np.sum(val_lls)/len(val_lls)


def compute_likelihood(predictions, count_first_prob=False):
    epsilon=1e-07
    #np.clip(predictions, epsilon, 1-epsilon)
    lls=[]
    for i,pred in enumerate(predictions):
        sort_pred=pred[:]
        if not count_first_prob:
            sort_pred=sort_pred[1:]
        sort_pred=np.clip(sort_pred,epsilon,1.0-epsilon)
        if len(sort_pred)>0:
            lls.append(neg_log_likelihood(sort_pred)/len(sort_pred))

    return np.mean(lls)


def compute_unique_elements(seqs):
    unique_elems=set()
    for seq in seqs:
        for s in seq:
            unique_elems.add(s)
    return len(unique_elems)


def compute_seq_max_length(seqs):
    max_seq_length=0
    for seq in seqs:
        if len(seq)>max_seq_length:
            max_seq_length=len(seq)
    return max_seq_length

def sample_weights(alpha,sigma):
    samples=np.zeros(alpha.shape)
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            samples[i,j]=np.random.normal(alpha[i,j], sigma, 1)
    return samples