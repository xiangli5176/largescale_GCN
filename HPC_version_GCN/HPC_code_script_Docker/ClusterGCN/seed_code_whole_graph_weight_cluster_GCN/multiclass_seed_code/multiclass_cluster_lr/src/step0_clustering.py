
# import torch

from multi_exec_code import *



if __name__ == '__main__':
    current_path = os.environ.get('PBS_O_WORKDIR').split('/')
    data_name = current_path[-2]
    test_folder_name = 'clusterGCN/'
    intermediate_data_folder = './'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name
    
    # this is the parts we divide the graph
    origin_train_batch_num = 32
    round_num = 1
    train_batch_num = round_num * origin_train_batch_num
   
    # # using the geometric package dataset:
    # from torch_geometric.datasets import Reddit
    # local_data_root = '~/GCN/Datasets/Geometric/'
    # dataset = Reddit(root = local_data_root + '/Reddit/')
    # data = dataset[0]
    # print_data_info(data, dataset)

    from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit, PPI_small
    remote_data_root = '~/GCN/Datasets/GraphSaint/'
    data_class = eval(data_name)
    dataset = data_class(root = remote_data_root + data_name)
    data = dataset[0]
    print_data_info(data, dataset)


    # from torch_geometric.datasets import Planetoid
    # local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    # dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    # data = dataset[0]


    step0_generate_clustering_machine(data, dataset, intermediate_data_folder, origin_train_batch_num, validation_ratio = 0.25, test_ratio = 0.25, mini_cluster_num = 128, round_num = round_num)
