import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import shutil
import copy
import csv

import pandas as pd
import seaborn as sns

# from Cluster_Machine import ClusteringMachine
# from Cluster_Trainer import ClusterGCNTrainer_mini_Train
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops


def check_folder_exist(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

def print_dir_content_info(path):
    """
        print out the file information under the path :  (name, size(KB))
    """
    with os.scandir(path) as dir_contents:
        print('\n Information about the content of ' + path)
        for entry in dir_contents:
            if entry.is_file():
                info = entry.stat()
                
                print('File name: [ {} ]; with size: {} KB'.format(entry.name, info.st_size / 1024))

        print()

def output_GPU_memory_usage(file_name, target_folder, comment =''):
    os.makedirs(os.path.dirname(target_folder), exist_ok=True)
    target_file = target_folder + file_name

    with open(target_file, 'a', newline='\n') as fp:
        fp.write('\n')
        fp.write(comment + '\n')
        fp.write('=' * 80 + '\nWith the Pytorch Version: ' )
        fp.write(torch.__version__ + '\n')
        fp.write('-' * 30 + 'info about allocated memory: ' + '-' * 30 + '\n')
        fp.write('GPU allocated memory is: {}\n'.format(torch.cuda.memory_allocated()))
        fp.write('GPU max_memory_allocated in bytes: {}\n'.format(torch.cuda.max_memory_allocated()))
        torch.cuda.reset_max_memory_allocated()

        fp.write('-' * 30 + 'info about cached memory: ' + '\n')
        fp.write('GPU memory_cached in bytes: {}\n'.format(torch.cuda.memory_cached()))
        fp.write('GPU max_memory_cached in bytes: {}\n'.format(torch.cuda.max_memory_cached()))
        torch.cuda.reset_max_memory_cached()
        
# preprocessing the data: 
# generate the edge_weight after adding self-loops
def get_edge_weight(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None, store_path='./tmp/'):
    """
        edge_index(ndarray): undirected edge index (two-directions both included)
        num_nodes(int):  number of nodes inside the graph
        edge_weight(ndarray): if any weights already assigned, otherwise will be generated 
        improved(boolean):   may assign 2 to the self loop weight if true
        store_path(string): the path of the folder to contain all the clustering information files
    """
    # calculate the global graph properties, global edge weights
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    fill_value = 1 if not improved else 2
    # there are num_nodes self-loop edges added after the edge_index
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index   
    # row includes the starting points of the edges  (first row of edge_index)
    # col includes the ending points of the edges   (second row of edge_index)

    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # row records the source nodes, which is the index we are trying to add
    # deg will record the out-degree of each node of x_i in all edges (x_i, x_j) including self_loops

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # normalize the edge weight
    normalized_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # transfer from tensor to the numpy to construct the dict for the edge_weights
    edge_index = edge_index.t().numpy()
    normalized_edge_weight = normalized_edge_weight.numpy()

    num_edge = edge_index.shape[0]

    output = ([edge_index[i][0], edge_index[i][1], normalized_edge_weight[i]] for i in range(num_edge))

    # output the edge weights as the csv file
    input_edge_weight_txt_file = store_path + 'input_edge_weight_list.csv'
    os.makedirs(os.path.dirname(input_edge_weight_txt_file), exist_ok=True)
    with open(input_edge_weight_txt_file, 'w', newline='\n') as fp:
        wr = csv.writer(fp, delimiter = ' ')
        for line in output:
            wr.writerow(line)
    return input_edge_weight_txt_file

# remove all the isolated nodes, otherwise metis won't work normally
def filter_out_isolate(edge_index, features, label):
    """
    edge_index: torch.Tensor (2 by 2N) for undirected edges in COO format
    features:  torch.Tensor(N by k)  for N nodes and K features
    label: torch.Tensor  (N, )  classifying labels for N nodes
    """
    edge_index_list = edge_index.t().numpy().tolist()
    connect_graph = nx.from_edgelist(edge_index_list)
    # filter out all the isolated nodes:
    connected_nodes_idx = sorted(node for node in connect_graph.nodes())
    # if the connected nodes is less than the total graph nodes
    if len(connected_nodes_idx) < features.shape[0]:
    #     print(edge_index.shape, type(edge_index))

        mapper = {node: i for i, node in enumerate(connected_nodes_idx)}
        connect_edge_index = [ [ mapper[edge[0]], mapper[edge[1]] ] for edge in edge_index_list ] 
    #     print(len(connected_nodes_idx), connected_nodes_idx[0], connected_nodes_idx[-1])
    #     print(np.array(connect_edge_index) )
        connect_edge_index =  torch.from_numpy(np.array(connect_edge_index)).t()
    #     print(connect_edge_index.shape)

        connect_features = features[connected_nodes_idx, :]
    #     print(connect_features.shape, type(connect_features))

        connect_label = label[connected_nodes_idx]
    #     print(connect_label.shape, type(connect_label))
        return connect_edge_index, connect_features, connect_label
    else:
        return edge_index, features, label

''' Draw the information about the GCN calculating batch size '''
def draw_cluster_info(clustering_machine, data_name, img_path, comments = '_cluster_node_distr'):
    """
        Won't call this for mini-batch with no clustering 
    """
    cluster_id = clustering_machine.train_clusters    # a list of cluster indices
    cluster_datapoints = {'cluster_id': cluster_id,  \
                          'train_batch' : [clustering_machine.info_train_batch_size[idx] for idx in cluster_id], \
                          'cluster_size' : [clustering_machine.info_isolate_cluster_size[idx] for idx in cluster_id], \
                         }
                         
    df = pd.DataFrame(data=cluster_datapoints, dtype=np.int32)
    # print(df)
    df_reshape = df.melt('cluster_id', var_name = 'clusters', value_name = 'node_num')
    
    plt.clf()
    plt.figure()
    sns.set(style='whitegrid')
    g = sns.catplot(x="cluster_id", y="node_num", hue='clusters', kind='bar', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + comments)
    g.set_xlabels("Cluster ID")
    g.set_ylabels("Number of nodes")
    
    img_name = img_path + data_name + comments
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    g.savefig(img_name, bbox_inches='tight')


class draw_trainer_info:
    
    def __init__(self, data_name, ClusterGCNTrainer, image_save_path, comments):
        self.data_name = data_name
        self.image_save_path = image_save_path
        self.comments = comments
        epoch_id = list(range(len(ClusterGCNTrainer.record_ave_training_loss)))
        self.trainer_data = {'epoch_id': epoch_id,  \
                             'ave_loss_per_node' : ClusterGCNTrainer.record_ave_training_loss \
                             }

        self.df = pd.DataFrame(data = self.trainer_data, dtype=np.float64)
    
    def draw_ave_loss_per_node(self):
        plt.clf()
        plt.figure()
        sns.set(style='whitegrid')
        g = sns.lineplot(x="epoch_id", y="ave_loss_per_node", data = self.df)
        g.set_title(self.data_name + ' Ave training loss vs epoch ' + self.comments)
        g.set(xlabel='epoch ID', ylabel='Ave training loss per node')
#         fig = g.get_figure()
        filename = self.image_save_path + self.data_name + '_train_loss_' + self.comments
        # estalbish a new directory for the target saving file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        


''' func for Execution of each specific model '''
def store_data_multi_tests(f1_data, data_name, graph_model, img_path, comments):
    run_id = sorted(f1_data.keys())
    run_data = {'run_id': run_id}
    
    run_data.update({model_name : [f1_data[key][idx] for key in run_id] for idx, model_name in enumerate(graph_model)})
    
    pickle_filename = img_path + data_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename


''' func for tuning hyper-parameter of the mini-batch model '''
def store_data_multi_tuning(tune_params, target, data_name, img_path, comments):
    """
        tune_params: is the tuning parameter list
        target: is the result, here should be F1-score, accuraycy, load time, train time
    """
    run_ids = sorted(target.keys())   # key is the run_id
    run_data = {'run_id': run_ids}
    # the key can be converted to string or not: i.e. str(tune_val)
    # here we keep it as integer such that we want it to follow order
    tmp = {tune_val : [target[run_id][tune_val] for run_id in run_ids] for tune_val in tune_params}  # the value is list
    run_data.update(tmp)
    
    pickle_filename = img_path + data_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename

def store_data_multi_investigate(investigate_res, data_name, res_name, img_path, comments):
    """
        investigate_res: currently either F1-score or accuracy a dict {epoch num : value}
    """
    run_id = sorted(investigate_res.keys())
    run_data = {'run_id': run_id}
    
    epoch_num_range = sorted(investigate_res[0].keys())  # at least one entry exists inside the dictionary and the epoch range is fixed
    run_data.update({epoch_num : [investigate_res[key][epoch_num] for key in run_id] for epoch_num in epoch_num_range})
    
    pickle_filename = img_path + data_name + '_' + res_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename


def draw_data_multi_tests(pickle_filename, data_name, comments, xlabel, ylabel):
    """
        Draw the figure for from the stored data (multiple store functions)
    """
    df = pd.read_pickle(pickle_filename)
    df_reshape = df.melt('run_id', var_name = 'model', value_name = ylabel)

    plt.clf()
    plt.figure()
    sns.set(style='whitegrid')
    g = sns.catplot(x="model", y=ylabel, kind='box', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + ' ' + ylabel + ' ' + comments)
    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    img_name = pickle_filename[:-4] + '_img'
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    plt.savefig(img_name, bbox_inches='tight')