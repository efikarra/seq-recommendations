'''
Created on 5 Apr 2017

@author: efi
'''
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np

class Preprocessor():
    def __init__(self,vocab,cs_vocab=None,pad_value=0.,sparse=True,seq_length=None):
        self.vocab=vocab
        self.cs_vocab=cs_vocab
        self.seq_length=seq_length
        self.pad_value=pad_value
        self.sparse=sparse
    def _pad_sequences(self,sequences):
        padded_seqs=pad_sequences(sequences,maxlen=self.seq_length,dtype='int32',padding='pre',truncating='pre',value=self.pad_value)
        self.seq_length=padded_seqs.shape[1]
        return padded_seqs.tolist()
    def transform_data(self,sequences,cs=None,xs=None):
        pass
class BaselinePreprocessor(Preprocessor):
    def __init__(self,vocab,cs_vocab=None,pad_value=0.,sparse=True,seq_length=None):
        Preprocessor.__init__(self, vocab, cs_vocab, pad_value, sparse, seq_length)
    def transform_data(self,sequences,cs=None,xs=None):
        #sequences=sequences[:]
        X_data=[]
        Y_data=[]
        index=0
        for seq in sequences:
            x_seq=[]
            y_seq=[]
            for i in range(0, len(seq)-1, 1):
                features=[]
                targets=[]
                if self.sparse:
                    features.append(seq[i]+1)
                    targets.append(seq[i+1]+1)
                else:
                    features+=np_utils.to_categorical([self.vocab[seq[i]]],len(self.vocab))[0,:].tolist()
                    targets+=np_utils.to_categorical([self.vocab[seq[i+1]]],len(self.vocab))[0,:].tolist()
                if xs:
                    features+=xs[index][i]
                if cs!=None and self.cs_vocab:
                    if self.cs_vocab:
                        features+=np_utils.to_categorical([self.cs_vocab[cs[index][i]]],len(self.cs_vocab))[0,:].tolist()
                x_seq.append(features)
                y_seq.append(targets)
            X_data.append(x_seq)
            Y_data.append(y_seq)
            index+=1
        features_dim=len(self.vocab) 
        if xs: 
            features_dim+=len(self.vocab)  
        if cs!=None and self.cs_vocab:
            features_dim+=len(self.cs_vocab)
        padded_X_data=self._pad_sequences(X_data)
        padded_Y_data=self._pad_sequences(Y_data)
        if self.sparse:
            return np.reshape(padded_X_data,(len(padded_X_data),self.seq_length,1)),np.reshape(padded_Y_data,(len(padded_X_data),self.seq_length,1))
        else:
            return np.reshape(padded_X_data,(len(padded_X_data),self.seq_length,features_dim)),np.reshape(padded_Y_data,(len(padded_Y_data),self.seq_length,len(self.vocab)))
class FullModelPreprocessor(Preprocessor):
    def __init__(self,vocab,cs_vocab=None,pad_value=0.,sparse=True,seq_length=None):
        Preprocessor.__init__(self, vocab, cs_vocab, pad_value, sparse, seq_length)
    def transform_data(self,sequences,xs,cs=None):
        #sequences=sequences[:]
        X_data=[]
        Context_Data=[]
        Y_data=[]
        for xi in xs:
            x_seq=[]
            for i in range(0, len(xi)-1, 1):
                x_seq.append(xi[i])
            Context_Data.append(x_seq)
            
        for j,seq in enumerate(sequences):
            x_seq=[]
            y_seq=[]
            for i in range(0, len(seq)-1, 1):
                features=[]
                targets=[]
                if self.sparse:
                    features.append(seq[i]+1)
                    targets.append(seq[i+1]+1)
                else:
                    features+=np_utils.to_categorical([self.vocab[seq[i]]],len(self.vocab))[0,:].tolist()
                    targets+=np_utils.to_categorical([self.vocab[seq[i+1]]],len(self.vocab))[0,:].tolist()
                x_seq.append(features)
                y_seq.append(targets)
            X_data.append(x_seq)
            Y_data.append(y_seq)
        features_dim=len(self.vocab) 
        padded_X_data=self._pad_sequences(X_data)
        padded_Context_data=self._pad_sequences(Context_Data)
        padded_Y_data=self._pad_sequences(Y_data)
        if self.sparse:
            return np.reshape(padded_X_data,(len(padded_X_data),self.seq_length,1)),np.reshape(padded_Y_data,(len(padded_X_data),self.seq_length,1)),np.reshape(padded_Context_data,(len(padded_Context_data),self.seq_length,features_dim))
        else:
            return np.reshape(padded_X_data,(len(padded_X_data),self.seq_length,features_dim)),np.reshape(padded_Y_data,(len(padded_Y_data),self.seq_length,len(self.vocab))),np.reshape(padded_Context_data,(len(padded_Context_data),self.seq_length,features_dim))
        