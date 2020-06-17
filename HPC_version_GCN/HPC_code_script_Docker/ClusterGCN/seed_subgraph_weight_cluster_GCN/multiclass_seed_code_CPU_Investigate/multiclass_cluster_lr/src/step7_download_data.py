
# import torch

from multi_exec_code import *



if __name__ == '__main__':
    current_path = os.environ.get('PBS_O_WORKDIR').split('/')
    data_name = current_path[-2]
    test_folder_name = 'clusterGCN/'
    intermediate_data_folder = './'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name
    
    # this is the parts we divide the graph
    origin_train_batch_num = 75
    GCN_layer = [128]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num
    
    tune_param_name = 'learning_rate'
    tune_val_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    trainer_list = list(range(7))
    round_num = 1
    train_batch_num = round_num * origin_train_batch_num

    from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit
    remote_data_root = './GCN/Datasets/GraphSaint/'
    data_class = eval(data_name)
    dataset = data_class(root = remote_data_root + data_name)
    data = dataset[0]
    print_data_info(data, dataset)
