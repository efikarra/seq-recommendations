'''
Created on 12 Apr 2017

@author: efi
'''
from main.datasets import *

if __name__ == "__main__":
    #load flickr DataFrame
    flickr_data=load_flickr_data()
    #get statistics for the whole dataset
    print "all data statistics:"
    print flickr_table_statistics(flickr_data)
    print
    #remove flickr sequences (i.e users) with length less than min_seq_length (if you want).
    #flickr_data=clean_flickr_data(flickr_data,min_seq_length=1)
    
    #Most likely option 1. is enough!
    
    #1. You can directly split the flickr data frame into train/val/test data.
    #Set test_percent=0.0 if you only want to split into train/val sets.
    train_table,val_table,test_table=split_flickr_train_val_table(flickr_data,train_percent=0.7,val_percent=0.3,test_percent=0.0)
    seqs_train,seqs_val,seqs_test,xs_train,xs_val,xs_test,vocab=build_flickr_train_val_seqs(train_table,val_table,test_table)
    #get statistics for the train/test sets
    print
    print "train statistics:"
    print flickr_table_statistics(train_table)
    print
    print "val statistics:"
    print flickr_table_statistics(val_table)
    
    #2. Or you can extract the whole sequences from flickr data frame and then you can split the sequences into train/val/test sequences.
    #Maybe useless but, if xs=True you get the xs for each sequence before splitting into train/val/test.
    sequences,vocab,total_xs=build_flickr_sequences(flickr_data,xs=True)
    seqs_train,seqs_val,seqs_test,xs_train,xs_val,xs_test=split_flickr_train_val_seqs(sequences,vocab,train_percent=0.7,val_percent=0.3,test_percent=0.0)