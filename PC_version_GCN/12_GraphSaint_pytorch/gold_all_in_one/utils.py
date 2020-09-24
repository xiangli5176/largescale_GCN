import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml
import scipy.sparse as sp

import csv
from torch.autograd import Variable


def to_numpy(x):
    """
        The original purpose of Variables was to be able to use automatic differentiation
        Autograd automatically supports Tensors with requires_grad set to True
    """
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def generate_tuning_raw_data_table(data_dict, file_path, file_name, tune_param_name):
    """
        data_dict : a dictionary of different runing index with different tuning values
                data_dict[1]: e.g.  index 1 runing, this is a dictionary of tuning values
    """
    target_file = file_path + file_name
    with open(target_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        header = [tune_param_name] + list(data_dict[0].keys())
        wr.writerow(header)
        for i, tune_val in data_dict.items():
            tmp_line = [i] + [tune_val[key] for key in tune_val.keys()]
            wr.writerow(tmp_line)


def load_data(prefix, datapath = './', normalize=True):
    """
        prefix: should be the dataname: flickr, PPI_small, Reddit, Yelp, PPI_large
        datapath: location for all dataset files
    """
    # Load a sparse matrix from a file using .npz format. Return csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix
    adj_full = scipy.sparse.load_npz(datapath + '{}/raw/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = scipy.sparse.load_npz(datapath + '{}/raw/adj_train.npz'.format(prefix)).astype(np.bool)
    
    role = json.load(open(datapath + '{}/raw/role.json'.format(prefix)))

    # print("role keys are: ", role.keys())
    
    """
        .npy:  the standard binary file format in NumPy for persisting a single arbitrary NumPy array on disk.
        .npz:  simple way to combine multiple arrays into a single file, one can use ZipFile to contain multiple “.npy” files

        .npz is just a ZipFile containing multiple “.npy” files. 
        And this ZipFile can be either compressed (by using np.savez_compressed) or uncompressed (by using np.savez)
    """
    # Load arrays or pickled objects from .npy, .npz or pickled files.
    feats = np.load(datapath + '{}/raw/feats.npy'.format(prefix))
    """
        json.load() method (without “s” in “load”) used to read JSON encoded data from a file and convert it into Python dictionary.
        json.loads() method, which is used for parse valid JSON String into Python dictionary
    """
    class_map = json.load(open(datapath + '{}/raw/class_map.json'.format(prefix)))
    class_map = {int(k):v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    # scipy.sparse.csr_matrix.nonzero:  Returns a tuple of arrays (row,col) containing the indices of the non-zero elements of the matrix.
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    # transform the whole feature by fitting the train features 
    feats = scaler.transform(feats)
    # -------------------------
    # adj_full, adj_train: csr_matrix ; feats: np.array;  class_map, role : python dict
    
    return adj_full, adj_train, feats, class_map, role

def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    Input:
        adj_full, adj_train: csr_matrix
        feats: np.array
        class_map  : python dict;  value can be a list (multi-label task), or can be a value: (multi-class task)
        role : python dict
    Output:
        Return: mainly get the class array as np.array
        
    """
    num_vertices = adj_full.shape[0]
    # check whether it is multi-class or multi-label task
    if isinstance(list(class_map.values())[0], list):   # one node belongs to multi-labels
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v            # assign a list directly to a row of numpy array
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1   # assume all the classes are continuous
        class_arr = np.zeros((num_vertices, num_classes))
        # if multi-class task: then shift the class label value, starting from 0
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role


def parse_layer_yml(arch_gcn, dim_input):
    """
        arch_gcn (dict of network structure): architecture of GCN 
    """
    num_layers = len(arch_gcn['arch'].split('-'))
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']] * num_layers
    act_layer = [arch_gcn['act']] * num_layers
    aggr_layer = [arch_gcn['aggr']] * num_layers
    dims_layer = [arch_gcn['dim']] * num_layers
    
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    
    return [dim_input] + dims_layer, order_layer, act_layer, bias_layer, aggr_layer




# mark for global: args_global.dir_log
def log_dir(f_train_config, prefix, git_branch, git_rev,timestamp, dir_log):
    """
    keep a record to the log files 
    Input:
        dir_log: from the globals, indicaet the log file path
        prefix: path of the dataset
        f_train_config: configuration file for the training
    """
    import getpass   # may be used by some func call inside here?
#     getpass.getpass([prompt[, stream]])
#     Prompt the user for a password without echoing
    
    log_dir = dir_log + "/log_train/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir

def sess_dir(dims, train_config, prefix, git_branch, git_rev, timestamp):
    """
    Inputs:
        
    """
    import getpass
    log_dir = "saved_models/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model = 'graphsaint',
            gitrev = git_rev.strip(),
            layer = '-'.join(dims),
            ts = timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    return sess_dir  # return the function object itself


def adj_norm(adj, deg = None, sort_indices = True):
    """
    Normalize adj according to two methods: symmetric normalization and rw normalization.
    sym norm is used in the original GCN paper (kipf)
    rw norm is used in graphsage and some other variants.

    # Procedure: 
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order rather than ascending order
    """
    diag_shape = (adj.shape[0], adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm

##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg, style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1 = _bcolors[style], msg = msg, color2='\033[0m'))


