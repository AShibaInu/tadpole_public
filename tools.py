"""Libraries"""

import torch
import math
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import nibabel as nib
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim.lr_scheduler as schd  
import torchvision.transforms as transforms

from torch.autograd import Variable


"""Preprocess"""

def preprocess(path, elim_sparse_feature = True, cutoff = 0.2):
    #path: path to the data folder
    #elim_sparse_feature: eliminate features that few people have. Percentange is based on the cutoff
    
    raw_data = pd.read_csv(path + "/TADPOLE_InputData.csv", low_memory=False)
    raw_data = raw_data.rename(columns = {'DX':'DX2'}) # Rename that column so there is no conflict in the algorithm

    """Obtain the IDs used for validation and test set"""
    val_id = pd.read_csv(path + "/TADPOLE_TargetData_test.csv", low_memory=False)
    test_id = pd.read_csv(path + "/TADPOLE_PredictTargetData_valid.csv", low_memory=False)

    val_id = val_id["PTID_Key"].unique().astype("int")
    test_id = test_id["PTID_Key"].unique().astype("int")

    """Columns 1888,1889, 1890 are numbers. Some number have inequality and some are wrongly typed as string
     The following lines convert everything to float and replace inqualities with NaN"""
    raw_data.iloc[:,1888] = pd.to_numeric(raw_data.iloc[:,1888],errors = "coerce")
    raw_data.iloc[:,1889] = pd.to_numeric(raw_data.iloc[:,1889],errors = "coerce")
    raw_data.iloc[:,1890] = pd.to_numeric(raw_data.iloc[:,1890],errors = "coerce")
    raw_data.iloc[:,3] = raw_data.iloc[:,3].astype('category')
    raw_data.iloc[:,6] = raw_data.iloc[:,6].astype('category')
    raw_data.iloc[:,10] = raw_data.iloc[:,10].astype('category')

    """ Column 1891 is mixture of string and number. Get rid of the string """
    raw_data.iloc[:,1891] = raw_data.iloc[:,1891].str.extract('(\d+)', expand = False)
    raw_data.iloc[:,1891] = pd.to_numeric(raw_data.iloc[:,1891])

    """Remove Columns that does not need one-hot ecoding"""
    n_row = raw_data.shape[0]
    n_col = raw_data.shape[1]
    removed_col = [] 
    for n in range(n_col):
        if ("PTID_Key" in raw_data.columns[n] or
            "EXAMDATE" in raw_data.columns[n] or 
            "VERSION" in raw_data.columns[n] or 
            "update" in raw_data.columns[n] or 
            "RUNDATE" in raw_data.columns[n] or 
            "STATUS" in raw_data.columns[n] or 
            "BATCH_UPENNBIOMK9_04_19_17" in raw_data.columns[n] or 
            "KIT_UPENNBIOMK9_04_19_17" in raw_data.columns[n] or 
            "STDS_UPENNBIOMK9_04_19_17" in raw_data.columns[n]) :
            removed_col += [n]    

    n = np.arange(n_col)
    n = np.setxor1d(removed_col,n)
    data = raw_data.iloc[:,n]

    """Search for categorical column and store where NaN are located"""
    categorical_col = []
    for c in range(data.shape[1]):
        if ((str(data.iloc[:,c].dtype)) == str("category") or 
            data.iloc[:,c].dtype is np.dtype('O')):
            categorical_col += [data.columns[c]]

    """One-hot encode"""
    _nan_categories = data.isnull()
    data = pd.get_dummies(data)

    # Put NaN for the categorical data
    for name in categorical_col:
        data.loc[_nan_categories[name], data.columns.str.startswith(name)] = np.NaN

    """Find the Location of NaN. It will be used for the indicator function"""
    indicators = pd.isnull(data)
    # 1 for existing data and 0 to nan
    indicators = (indicators*1==0)*1

    """Reattach removed columns at the end"""
    data = pd.concat([data, raw_data.iloc[:,removed_col]], axis=1)
    indicators = pd.concat([indicators, raw_data.iloc[:,removed_col]], axis=1) 
    #Only ID is needed for the indicators
    
    """Replace -4 with NaN"""
    data = data.replace(-4, np.NaN)
    #Make it look pretty :) 
    data = data.replace(np.nan,np.NaN)
    data = data.replace(np.NAN,np.NaN)

    """Separate Train, Val, Test"""
    groups = [data for _, data in data.groupby("PTID_Key")]
    indicators_group = [indicators for _, indicators in indicators.groupby("PTID_Key")]

    trn_indicator = []
    val_indicator = []
    test_indicator = []
    xTrain = []
    xVal = []
    xTest = []
    for n in range(len(groups)):
        subject = groups[n]["PTID_Key"].unique()[0].astype("int")
        if np.any(val_id == subject):
            groups[n]["EXAMDATE"] = pd.to_datetime(groups[n]["EXAMDATE"])
            groups[n] = groups[n].sort_values("EXAMDATE")
            indicators_group[n]["EXAMDATE"] = pd.to_datetime(indicators_group[n]["EXAMDATE"])
            indicators_group[n] = indicators_group[n].sort_values("EXAMDATE")                       
            xVal += [groups[n]]
            val_indicator += [indicators_group[n]]
        elif np.any(test_id == subject):
            groups[n]["EXAMDATE"] = pd.to_datetime(groups[n]["EXAMDATE"])
            groups[n] = groups[n].sort_values("EXAMDATE")
            indicators_group[n]["EXAMDATE"] = pd.to_datetime(indicators_group[n]["EXAMDATE"])
            indicators_group[n] = indicators_group[n].sort_values("EXAMDATE")  
            xTest += [groups[n]]
            test_indicator += [indicators_group[n]]
        else:
            groups[n]["EXAMDATE"] = pd.to_datetime(groups[n]["EXAMDATE"])
            groups[n] = groups[n].sort_values("EXAMDATE")
            indicators_group[n]["EXAMDATE"] = pd.to_datetime(indicators_group[n]["EXAMDATE"])
            indicators_group[n] = indicators_group[n].sort_values("EXAMDATE")  
            xTrain += [groups[n]]
            trn_indicator += [indicators_group[n]]

    xTrain = pd.concat(xTrain).reset_index(drop=True)
    xVal = pd.concat(xVal).reset_index(drop=True)
    xTest = pd.concat(xTest).reset_index(drop=True)

    trn_indicator = pd.concat(trn_indicator).reset_index(drop=True)
    val_indicator = pd.concat(val_indicator).reset_index(drop=True)
    test_indicator = pd.concat(test_indicator).reset_index(drop=True)

    """Standarize the Features. Only xTrain is used to calculate the mean
    NaN replace by one."""
    trn_mean = []
    trn_std = []
    xTrain_norm = xTrain
    xVal_norm = xVal
    xTest_norm = xTest
    for c in range(1833):                   
        mean = xTrain.iloc[:,c].mean()
        std = xTrain.iloc[:,c].std()
        xTrain_norm.iloc[:,c] = (xTrain.iloc[:,c]-mean)/std
        xVal_norm.iloc[:,c] = (xVal.iloc[:,c]-mean)/std
        xTest_norm.iloc[:,c] = (xTest.iloc[:,c]-mean)/std
        trn_mean += [mean]
        trn_std += [std]
    
    #Interpolate missing value. If the first value is missing, backfilled it
    xTrain_norm.iloc[:,:1833] = xTrain_norm.iloc[:,:1833].interpolate(method = "linear").bfill()
    xVal_norm.iloc[:,:1833] = xVal_norm.iloc[:,:1833].interpolate(method = "linear").bfill()
    xTest_norm.iloc[:,:1833] = xTest_norm.iloc[:,:1833].interpolate(method = "linear").bfill()
    
    #Foward the One-hot encoding
    xTrain_norm.iloc[:,1833:] = xTrain_norm.iloc[:,1833:].ffill()
    xVal_norm.iloc[:,1833:] = xVal_norm.iloc[:,1833:].ffill()
    xTest_norm.iloc[:,1833:] = xTest_norm.iloc[:,1833:].ffill()  
    
    #If there is still missing value, fill it with 0 (the average)
    xTrain_norm.iloc[:,:1833] = xTrain_norm.iloc[:,:1833].replace(np.NaN, 0)
    xVal_norm.iloc[:,:1833] = xVal_norm.iloc[:,:1833].replace(np.NaN, 0)
    xTest_norm.iloc[:,:1833] = xTest_norm.iloc[:,:1833].replace(np.NaN, 0)

    # For categorical data, NAN = 0
    xTrain_norm.iloc[:,1833:] = xTrain_norm.iloc[:,1833:].replace(np.NaN, 0)
    xVal_norm.iloc[:,1833:] = xVal_norm.iloc[:,1833:].replace(np.NaN, 0)
    xTest_norm.iloc[:,1833:] = xTest_norm.iloc[:,1833:].replace(np.NaN, 0)

    """Note we need only the first 1943 colums
    everything else is nonrelevant (dates,update,etc)"""
    # The last column has the IDs
    xTrain_norm = xTrain_norm.iloc[:,:1944]
    xVal_norm = xVal_norm.iloc[:,:1944]
    xTest_norm = xTest_norm.iloc[:,:1944]

    trn_indicator = trn_indicator.iloc[:,:1944]
    val_indicator = val_indicator.iloc[:,:1944]
    test_indicator = test_indicator.iloc[:,:1944]

    """Eliminate Feature that few people have """
    if elim_sparse_feature:
        percent_filled = np.sum(trn_indicator.as_matrix(), axis = 0)/trn_indicator.shape[0] 
        #percentage of the column that have data
        keep_col = percent_filled > cutoff
        keep_col_inx = np.argwhere(keep_col).flatten()

        xTrain_norm = xTrain_norm.iloc[:,keep_col_inx]
        xVal_norm = xVal_norm.iloc[:,keep_col_inx]
        xTest_norm = xTest_norm.iloc[:,keep_col_inx]

        trn_indicator = trn_indicator.iloc[:,keep_col_inx]
        val_indicator = val_indicator.iloc[:,keep_col_inx]
        test_indicator = test_indicator.iloc[:,keep_col_inx]
        
    return xTrain_norm, xVal_norm, xTest_norm, trn_indicator, val_indicator, test_indicator

"""Pad/Mask Sequence"""

def make_list(path, xTrain, xVal, xTest):
    #convert Pandas object into a list where each element in the list is a subject
    
    """Train"""
    #Make sure that the xTrain have the same number of subject that yTrain
    trn_id = pd.read_csv(path + "/TADPOLE_TargetData_train.csv", low_memory=False)
    trn_id = trn_id["PTID_Key"].unique().astype("int")
    xtrain = [xTrain for _, xTrain in xTrain.groupby("PTID_Key")]
    
    xTrain_list = []
    for n in range(len(xtrain)):
        subject = xtrain[n]["PTID_Key"].unique()[0].astype("int")
        if np.any(trn_id == subject):             
            xTrain_list += [xtrain[n]]
    
    """Val"""
    xVal_list = [xVal for _, xVal in xVal.groupby("PTID_Key")]
    
    """Test"""
    xTest_list = [xTest for _, xTest in xTest.groupby("PTID_Key")]
    
    return xTrain_list, xVal_list, xTest_list



def get_matrix(samples):
    #Transform Pandas object to matrix and eliminate the last column
    subjects = []
    for n in samples:
        subjects += [n.iloc[:,:].as_matrix()]
    return subjects
 
def get_pad(samples, tensor=True):
    sizes = [len(seq) for seq in samples]
    max_size = max(sizes)

    padded_seq = []
    len_seq = []
    for seqs in samples:
        len_seq += [seqs.shape[0]]
        new_seq = np.zeros((max_size,seqs.shape[1]))
        new_seq[:seqs.shape[0],:] = seqs
        padded_seq += [new_seq]
    
    padded_seq = np.stack(padded_seq)
    return padded_seq, np.asarray(len_seq)

def get_first_alz_diag(target):
    
    # Get first alzeihmer diagnosis
    alz_status = []
    delete_n = []
    for n in range(len(target)):
        target[n]["Date"] = pd.to_datetime(target[n]["Date"])
        target[n] = target[n].sort_values("Date")
        target[n] = target[n].bfill()
        status = target[n].iloc[0,2:5].argmax()
        
        if status == 'CN_Diag':
            alz_status += [0]
        elif status == 'MCI_Diag':
            alz_status += [1]
        elif status == 'AD_Diag':
            alz_status += [2]
        else:
            alz_status += [3]
            delete_n += [n]
    
    alz_status = np.stack(alz_status)
    return alz_status, np.asarray(delete_n)

def make_batch(batch_size, x, y, x_sequence, y_sequence):
    batch_idx = np.random.choice(x.shape[0],batch_size)
    
    x_batch = x[batch_idx]
    y_batch = y[batch_idx]
    x_sequence_batch = x_sequence[batch_idx]
    y_sequence_batch = y_sequence[batch_idx]
    
    sorted_idx = np.flip(np.argsort(x_sequence_batch),0)
    # Needs to sort sequence length for pad-padded sequence to work in the input
    x_sorted = x_batch[sorted_idx]
    y_sorted = y_batch[sorted_idx]
    x_sequence_sorted = x_sequence_batch[sorted_idx]
    y_sequence_sorted = y_sequence_batch[sorted_idx]
    
    return x_sorted, y_sorted, x_sequence_sorted.tolist(), y_sequence_sorted.tolist()

def validate(inputs,targets,sequence, encoder, decoder, criterion):
    
    running_loss = 0.0
    correct = 0
    total = 0        
    for i in range(20):
        # Training
        encoder.eval()
        decoder.eval()
        
        #Mini-batch
        x, y, seq = make_batch(32, inputs, targets, sequence)
        
        x = Variable(torch.from_numpy(x[:,:,:-1]).float().cuda())
        y = Variable(torch.from_numpy(y).long().cuda())

        # Forward + Backward + Optimize
        hidden = encoder(x,seq)
        init = Variable(torch.zeros(32,1,3).float().cuda())
        outputs = decoder(init, h0= hidden[1][0])
        loss = criterion(outputs, y)

        # Calculate statistics
        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted.view(-1) == y.data.view(-1)).sum()
        
    loss = running_loss/20
    accuracy = 100*correct/total
    
    return loss, accuracy

"""Resample the labels of a dataset"""

# A function that resamples the data given the datapath. If interpolation = True, 
# interpolation is performed and NaNs in the data are replaced. 
# Outputs two pandas dataframes, the first one corresponding to the resampled dataframe,
# the second one that keeps track of the original dates before resampling. If the datapoint
# is not present in the original dataset, original date is NaN. 
def resample_data(datapath, interpolation = True):
    
    # load data
    raw_data = pd.read_csv(datapath, low_memory=False)

    # Convert date to date time
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])

    # create a new column to keep track if the datapoint is resampled, and the original date
    raw_data["orig_date"] = 0

    group = [data for _, data in raw_data.groupby("PTID_Key")] #Group the same patient into a list
        
    # resample data
    
    resampled = []
    for patient in group:

        patient.sort_values(by='Date') # sort the date by date
        num_data = patient.shape[0]
        time_diff = patient["Date"].diff()
        time_diff = time_diff/ np.timedelta64(1,'M') #Convert to months
        num_resample = 1 # keep track of total number of resamples
        PTID_Key = patient.iloc[0,1]
        # resample the first data
        patient.iloc[0,-1] = patient.iloc[0,0]  # remember the orig date
        patient.iloc[0,0] = pd.to_datetime("2010-01-01") # resample date  

        for i in range(num_data-1): 

            resample_time = int( np.round(time_diff.iloc[i+1]/5) )

            # resample by (resample_time - 1) times            
            for j in range(resample_time-1):
                # add new points with NaNs except the date 
                new_date = pd.to_datetime("2010-01-01") + np.timedelta64(num_resample*5,'M')
                new = [new_date,PTID_Key,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
                patient.loc[len(patient)] = new
                num_resample += 1

            # resample the date of the existing point 
            patient.iloc[i+1,-1] = patient.iloc[i+1,0]  # remember the orig date
            patient.iloc[i+1,0] = pd.to_datetime("2010-01-01") + np.timedelta64(num_resample*5,'M') # resample date
            num_resample += 1     
    
        # sort again after resampling
        output = patient.sort_values(by='Date') 
        
        # interpolations
        if interpolation == True:

            output.iloc[:,5:8] = output.iloc[:,5:8].interpolate(method = "linear").bfill()
            output.iloc[:,5:8] = output.iloc[:,5:8].interpolate(method = "linear").ffill()
            output.iloc[:,2:5] = output.iloc[:,2:5].bfill()
            output.iloc[:,2:5] = output.iloc[:,2:5].ffill()

            #If there is still missing value, fill it with the average of raw data
            output.iloc[:,5:8] = output.iloc[:,5:8].fillna(raw_data.iloc[:,5:8].mean())
            # For categorical data, NAN = 0
            output.iloc[:,2:5] = output.iloc[:,2:5].replace(np.NaN, 0)
    
        resampled.append(output)
    
    resampled = pd.concat(resampled)
    
    return resampled.iloc[:,:8], resampled.iloc[:,-1]

"""Convert resampled dates back to orig dates"""

# Takes in resampled, orig_dates as inputs, and
# convert the resampled dates to orig dates, and 
# delete the resampled data. 
def get_origDates(resampled, orig_dates):
    resampled.iloc[:,0] = orig_dates
    orig = resampled.dropna()
    orig = orig.sort_index()
    
    return orig

"""Functions for targets"""

def get_average(data):
    """
    Input data is the resampled target.
    Returns the average and std of ADA, Ventricles and MMSE in tuple format
    """
    adas_mean = data["ADAS13"].mean()
    adas_std = data["ADAS13"].std()
    
    ventricle_mean = data["Ventricles_Norm"].mean()
    ventricle_std = data["Ventricles_Norm"].std()
    
    mmse_mean = data["MMSE"].mean()
    mmse_std = data["MMSE"].std()
    
    adas = (adas_mean, adas_std)
    ventricle = (ventricle_mean, ventricle_std)
    mmse = (mmse_mean, mmse_std)
    return adas, ventricle, mmse

def get_standarize(data, adas, ventricle, mmse):
    """
    Input data is the resampled target.
    adas, ventricle and mmse must be tuples of the form (mean,std) from the training data
    Returns a standarized data frame
    Also replaces the timestamp to 0s, since is not important for prediction
    """
    data["ADAS13"] = (data["ADAS13"]-adas[0])/adas[1]
    data["Ventricles_Norm"] = (data["Ventricles_Norm"]-ventricle[0])/ventricle[1]
    data["MMSE"] = (data["MMSE"]-mmse[0])/mmse[1]
    data["Date"] = 0
    return data

def get_features(data, task = "classification"):
    """
    Input data is the resampled target.
    If the task is classifiacation: data only contain the columns for alz diagnosis
    If task is regression, it contains the data for adas, ventricle and mmse 
    """
    if task == "classification":
        data = data.rename(columns={'CN_Diag': '0CN_Diag', 'MCI_Diag': '1MCI_Diag', 'AD_Diag': '2AD_Diag'})
        data = data.iloc[:,1:5]
        code = data.iloc[:,1:].idxmax(axis=1).astype('category').cat.codes
        code.name = "DX"
        data = pd.concat([data,code], axis=1)
        
    elif task == "regression":
        data = data[["PTID_Key","ADAS13", "Ventricles_Norm", "MMSE"]]
        
    return data