import datasets
import utils

seqs, vocab = datasets.load_msnbc_data()
print 'MSNBC'
utils.self_transitions(seqs)

seqs, vocab = datasets.load_reddit_data()
print 'Reddit'
utils.self_transitions(seqs)

seqs, vocab = datasets.load_student_data()
print 'Student'
utils.self_transitions(seqs)

