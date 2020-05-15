from numpy import array
from numpy import reshape
from numpy import hstack
import math


def singlevar_split_sequence(sequence, memory_step):
    '''
    Split a univariate sequence into univariate samples.

    Input shape: sequence 1 dimention

    Output: 

    Feature vector of memory_step dimention vector

    Label vector of values, each value is label of the corresponding vector on Feature vector 
    '''
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + memory_step
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def multivar_split_sequence(sequence, memory_step):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + memory_step
        # check if we are beyond the dataset
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def parallel_split_sequence(sequence, memory_step):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + memory_step
        # check if we are beyond the dataset
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Reshape from [samples, timesteps] into [samples, timesteps, features]


def adaptive_input(X, num_features):
    return X.reshape(X.shape[0], X.shape[1], num_features)


def adaptive_input_cnn_lstm(X, memory_step, num_features, single_test=False):
    num_sub_sequences = int(math.sqrt(int(memory_step)))
    if single_test == True:
        return X.reshape(1, num_sub_sequences, num_sub_sequences, num_features)
    return X.reshape(X.shape[0], num_sub_sequences, num_sub_sequences, num_features)


def adaptive_input_conv_lstm(X, memory_step, num_features, single_test=False):
    num_sub_sequences = int(math.sqrt(int(memory_step)))
    if single_test == True:
        return X.reshape(1, num_sub_sequences, 1, num_sub_sequences, num_features)
    return X.reshape(X.shape[0], num_sub_sequences, 1, num_sub_sequences, num_features)


raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps = 3
X, y = multivar_split_sequence(raw_seq, n_steps)
print(X)
print(y)
# # summarize the data
# for i in range(len(X)):
# 	print(X[i], y[i])

# # [10 20 30] 40
# # [20 30 40] 50
# # [30 40 50] 60
# # [40 50 60] 70
# # [50 60 70] 80
# # [60 70 80] 90
