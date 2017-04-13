'''
Created on 4 Apr 2017

@author: efi
'''
import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import *
from model import *

def read_data(data_dir,filename,delimiter=",",skipinitialspace=True):
    data_path=os.path.join(data_dir, filename)
    data_table=pd.read_csv(data_path, delimiter=delimiter, parse_dates=[0], skipinitialspace=skipinitialspace)
    return data_table
def shift_ids(data_table,col,value=1):
    data_table[col] += 1
def group_by_user(data_table):
    grouped_by_user=data_table.sort_values(["userID","startTime"], ascending=True).groupby(["userID"])
    return grouped_by_user
def get_pois_byuser(grouped_by_user):
    grouped_pois=grouped_by_user["poiID"].apply(list).reset_index()["poiID"].values
    grouped_trajs=grouped_by_user["trajID"].apply(list).reset_index()["trajID"].values
    return grouped_pois,grouped_trajs

def get_pois_vocab(data_table):
    unique_pois=data_table["poiID"].unique()
    return dict(zip(unique_pois,range(len(unique_pois))))
def get_trajs_vocab(data_table):
    unique_trajs=data_table["trajID"].unique()
    return dict(zip(unique_trajs,range(len(unique_trajs))))
def pois_statistics(data_table):
    counts_by_user=data_table.groupby(["userID"])["poiID"].count().reset_index()
    basic_stats = pd.DataFrame([counts_by_user["poiID"].min(), counts_by_user["poiID"].max(), counts_by_user["poiID"].median(), counts_by_user["poiID"].mean()], \
                           index=['min','max', 'median', 'mean'])
    basic_stats.columns = ['seq_length']
    print basic_stats

def build_xs(grouped_pois,vocab):
    xs=[]
    for pois in grouped_pois:
        xi_s=[]
        xi=[0]*len(vocab)
        for poi in pois:
           xi[vocab[poi]]=1 
           xi_s.append(xi[:])
        xs.append(xi_s)
    return xs
def get_group_first_percentage(group,percent):
    elements=int(np.ceil(percent*len(group['poiID'])))
    return group.head(elements)
def get_group_last_percentage(group,percent):
    elements=int(np.floor(percent*len(group['poiID'])))
    return group.tail(elements)
def split_data(grouped_by_user):
    train_table=grouped_by_user.apply(get_group_first_percentage,0.7)
    test_val_table=grouped_by_user.apply(get_group_last_percentage,0.3)
#     val_table=test_val_table.groupby(["userID"]).apply(get_group_first_percentage,0.2)
#     test_table=test_val_table.groupby(["userID"]).apply(get_group_last_percentage,0.1)
    return train_table,test_val_table
def remove_short_sequences(data_table,min_seq_length):
    clean_table=data_table[data_table.groupby(["userID"])["poiID"].transform(len) > min_seq_length]
    return clean_table
def create_train_val_data(data_table,min_seq_length): 
    data_table=data_table.copy()
    data_table=remove_short_sequences(data_table,min_seq_length)
    grouped_by_user=group_by_user(data_table) 
    
    train_table,val_table=split_data(grouped_by_user)
#     shift_ids(data_table,"poiID",value=1)
#     shift_ids(data_table,"trajID",value=1)
    total_pois,total_trajs=get_pois_byuser(grouped_by_user)
    train_pois,train_trajs=get_pois_byuser(grouped_by_user)
    val_pois,val_trajs=get_pois_byuser(grouped_by_user)
    #test_pois,test_trajs=get_pois_byuser(test_table)
    
    
    print "total sequences:",len(train_pois)
    vocab=get_pois_vocab(train_table)
    #cs_vocab=get_trajs_vocab(train_table)
    print "vocabulary size:",len(vocab)
     
    xs_train=build_xs(train_pois,vocab)
    xs_val=build_xs(val_pois,vocab)
    #xs_test=build_xs(test_pois,vocab)
    return train_pois,val_pois,xs_train,xs_val,vocab    
def run_baseline(train_pois,val_pois,xs_train,xs_val,vocab,pathtosave="model/",filetoread=None):  
    preprocessor=BaselinePreprocessor(vocab=vocab,cs_vocab=None,pad_value=0.,sparse=False,seq_length=None)
    X_train,Y_train=preprocessor.transform_data(train_pois,xs=xs_train,cs=None)
    X_val,Y_val=preprocessor.transform_data(val_pois,xs=xs_val,cs=None)
    #X_test,Y_test=preprocessor.transform_data(train_pois,xs=xs,cs=None)
    print "(train sequences,train timesteps,input dimension):",X_train.shape
    print "(val sequences,val timesteps,val dimension):",X_val.shape
    baseline=RNNBaseline(X_train.shape[1], X_train.shape[2], len(vocab), model_name="baseline_model", rnn_type='LSTM',
                 loss='categorical_crossentropy', metrics=[], activation="relu", n_units=2)
    if not filetoread:
        baseline.fit_model(X_train, Y_train, X_val,Y_val,n_epochs=40, batch_size=10, verbose=1)
        baseline.save_model_weights(pathtosave)
    else:
        baseline.load_model_weights(filetoread)
    print baseline.get_layer_weights(2)
def run_fullmodel(train_pois,val_pois,xs_train,xs_val,vocab,pathtosave="model/",filetoread=None):  
    preprocessor=FullModelPreprocessor(vocab=vocab,cs_vocab=None,pad_value=0.,sparse=False,seq_length=None)
    X_train,Y_train,Xs_train=preprocessor.transform_data(train_pois,xs=xs_train)
    X_val,Y_val,Xs_val=preprocessor.transform_data(val_pois,xs=xs_val,cs=None)
    #X_test,Y_test=preprocessor.transform_data(train_pois,xs=xs,cs=None)
    print "(train sequences,train timesteps,input dimension):",X_train.shape
    print "(val sequences,val timesteps,val dimension):",X_val.shape
#     full_model=RNNBXModel(timesteps=X_train.shape[1], features=X_train.shape[2], n_classes=len(vocab), model_name="full_model", rnn_type='LSTM',
#                  loss='categorical_crossentropy', metrics=[], activation="relu", n_units=10,cv_features=Xs_train.shape[2],cv_units=Xs_train.shape[2],cv_activation="linear")
    full_model=RNNY2YModel(timesteps=X_train.shape[1], x_dim=X_train.shape[2], y_dim=len(vocab), model_name="full_model", rnn_type='simpleRNN',
                 loss='categorical_crossentropy', metrics=[], z_to_z_activation="relu", z_dim=10,y_to_y_activation="linear",output_bias=True)
#     if not filetoread:
#         full_model.fit_model([X_train,Xs_train], Y_train, [X_val,Xs_val],Y_val,n_epochs=20, batch_size=10, verbose=1)
#         full_model.save_model_weights(pathtosave)
#     else:
#         full_model.load_model_weights(filetoread)
    print baseline.get_layer_weights(2)
    
if __name__ == "__main__":
    data_table=read_data("data/","traj-noloop-all-Melb.csv",delimiter=",",skipinitialspace=True)
    pois_statistics(data_table)
    train_pois,val_pois,xs_train,xs_val,vocab=create_train_val_data(data_table,min_seq_length=6)
    #run_baseline(train_pois,val_pois,xs_train=xs_train,xs_val=xs_val,vocab=vocab,filetoread=None)
    #run_baseline(train_pois,val_pois,xs_train=xs_train,xs_val=xs_val,vocab=vocab,filetoread="model/baseline_model.h5")
    run_fullmodel(train_pois,val_pois,xs_train,xs_val,vocab)
    
    #full_model.fit_model([X_train,xs_train], Y_train, [X_val],Y_val,n_epochs=10, batch_size=10, verbose=1)
    
