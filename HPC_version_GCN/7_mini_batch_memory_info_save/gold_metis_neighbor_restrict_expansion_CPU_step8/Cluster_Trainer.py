import time
import torch
import random
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle

from Custom_GCNConv import Net

#####################################
###      Mini batch Trainer       ###
#####################################



class ClusterGCNTrainer_mini_Train(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, data_folder, in_channels, out_channels, input_layers = [32, 16], dropout=0.3):
        """
        :param in_channels, out_channels: input and output feature dimension
        :param clustering_machine:
        """  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_device = torch.device("cpu")
        
        self.data_folder = data_folder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_layers = input_layers
        self.dropout = dropout
        
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
#         print('used layers are: ', str(self.input_layers))
        self.model = Net(self.in_channels, self.out_channels, input_layers = self.input_layers, dropout = self.dropout)
        self.model = self.model.to(self.device)
    
    # call the forward function batch by batch
    def do_forward_pass(self, tr_train_nodes, tr_edges, tr_edge_weights, tr_features, tr_target):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        
        '''Target and features are one-one mapping'''
        # calculate the probabilites from log_sofmax
        predictions = self.model(tr_edges, tr_features, tr_edge_weights)
        
        ave_loss = torch.nn.functional.nll_loss(predictions[tr_train_nodes], tr_target[tr_train_nodes])
        node_count = tr_train_nodes.shape[0]

        # for each cluster keep track of the counts of the nodes
        return ave_loss, node_count


    def update_average_loss(self, batch_average_loss, node_count, isolate = True):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item() * node_count
        if isolate:
            self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss / self.node_count_seen
        return average_loss

    # iterate through epoch and also the clusters  
    def train_investigate_F1(self, epoch_num=10, learning_rate=0.01, weight_decay = 0.01, mini_epoch_num = 1, output_period = 10, train_batch_num = 2):
        """
            *** Periodically output the F1 score during training. After certain number of epochs ***
            epoch_num:  number of total training epoch number
            learning rate: learning rate during training
            weight_decay:  decay coefficients for the regularization
            mini_epoch_num:  number of epochs of repeating training after loading data on the GPU
            output_period:  number of epochs after which output the F1 and accuray to investigate the model refining process
        """
        # load the validation data for investigation
        batch_file_name = self.data_folder + 'validation/batch_whole'
        t00 = time.time()
        with open(batch_file_name, "rb") as fp:
            minibatch_data_validation = pickle.load(fp)
        read_time = (time.time() - t00) * 1000
        print('*** During validation for # {0} batch, reading batch file costed {1:.2f} ms ***'.format("whole graph for investigation", read_time) )
        investigate_f1 = {}
        investigate_accuracy = {}
        
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model.train()   #   set into train mode, only effective for certain modules such as dropout and batchNorm
        self.record_ave_training_loss = []
        self.time_train_load_data = 0
        
        epoch_partition = epoch_num // mini_epoch_num
        t0 = time.time()
        train_clusters = list(range(train_batch_num))
        for epoch_part in range(epoch_partition):
#             For test purpose, we let the clusters to follow specific order
            random.shuffle(train_clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in train_clusters:
                # for each batch, we load once and train it for multiple epochs:
                # read in the train data from the pickle files
                batch_file_name = self.data_folder + 'train/batch_' + str(cluster)
                
                t2 = time.time()
                with open(batch_file_name, "rb") as fp:
                    minibatch_data_train_group = pickle.load(fp)
                read_time = (time.time() - t2) * 1000
                print('*** During training for # {0:3d} batch, reading batch file costed {1:.2f} ms ***'.format(cluster, read_time) )
                # for each train batch: we need to go through each subdivided-mini_train_subset
                random.shuffle(minibatch_data_train_group)
                for minibatch_data_train in minibatch_data_train_group:
                    tr_train_nodes, tr_edges, tr_edge_weights, tr_features, tr_target = minibatch_data_train

                    t1 = time.time()
                    tr_train_nodes = tr_train_nodes.to(self.device)
                    tr_edges = tr_edges.to(self.device)
                    tr_edge_weights = tr_edge_weights.to(self.device)
                    tr_features = tr_features.to(self.device)
                    tr_target = tr_target.to(self.device)

                    self.time_train_load_data += (time.time() - t1) * 1000
                    # train each batch for multiple epochs
                    for mini_epoch in range(mini_epoch_num):
                        # record the current overall epoch index:
                        real_epoch_num = 1 + mini_epoch + mini_epoch_num * epoch_part # real_epoch_num starts from 0, therefore we add 1

                        self.optimizer.zero_grad()
                        batch_ave_loss, node_count = self.do_forward_pass(tr_train_nodes, tr_edges, tr_edge_weights, tr_features, tr_target)
                        batch_ave_loss.backward()
                        self.optimizer.step()
                        ave_loss = self.update_average_loss(batch_ave_loss, node_count)

                        # at this point finish a single train duration: update the parameter and calcualte the loss function
                        # periodically output the F1-score in the middle of the training process
                        if real_epoch_num % output_period == 0:
                            investigate_f1[real_epoch_num], investigate_accuracy[real_epoch_num] = self.middle_check_whole_cpu_validate(minibatch_data_validation)
#                             self.model.train()    # reset to the train mode
            
            self.record_ave_training_loss.append(ave_loss)
        # convert to ms
        self.time_train_total = ((time.time() - t0) * 1000)
        return investigate_f1, investigate_accuracy

    # iterate through epoch and also the clusters
    def train(self, epoch_num=10, learning_rate=0.01, weight_decay = 0.01, mini_epoch_num = 1, train_batch_num = 2):
        """
            *** Training a model. ***
            epoch_num:  number of total training epoch number
            learning rate: learning rate during training
            weight_decay:  decay coefficients for the regularization
            mini_epoch_num:  number of epochs of repeating training after loading data on the GPU
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model.train()
        self.record_ave_training_loss = []
        self.time_train_load_data = 0
        
        epoch_partition = epoch_num // mini_epoch_num
        t0 = time.time()
        train_clusters = list(range(train_batch_num))
        for epoch in range(epoch_partition):
#             For test purpose, we let the clusters to follow specific order
            random.shuffle(train_clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            
            for cluster in train_clusters:
                # read in the train data from the pickle files
                batch_file_name = self.data_folder + 'train/batch_' + str(cluster)
                
                t2 = time.time()
                with open(batch_file_name, "rb") as fp:
                    minibatch_data_train_group = pickle.load(fp)
                read_time = (time.time() - t2) * 1000
                print('*** During training for # {0:3d} batch, reading batch file costed {1:.2f} ms ***'.format(cluster, read_time) )
                # for each train batch: we need to go through each subdivided-mini_train_subset
                random.shuffle(minibatch_data_train_group)
                for minibatch_data_train in minibatch_data_train_group:
                    
                    tr_train_nodes, tr_edges, tr_edge_weights, tr_features, tr_target = minibatch_data_train

                    # for each cluster, we load once and train it for multiple epochs:
                    t1 = time.time()
                    tr_train_nodes = tr_train_nodes.to(self.device)
                    tr_edges = tr_edges.to(self.device)
                    tr_edge_weights = tr_edge_weights.to(self.device)
                    tr_features = tr_features.to(self.device)
                    tr_target = tr_target.to(self.device)


                    self.time_train_load_data += (time.time() - t1) * 1000
                    # train each batch for multiple epochs
                    for mini_epoch in range(mini_epoch_num):
                        self.optimizer.zero_grad()
                        batch_ave_loss, node_count = self.do_forward_pass(tr_train_nodes, tr_edges, tr_edge_weights, tr_features, tr_target)
                        batch_ave_loss.backward()
                        self.optimizer.step()
                        ave_loss = self.update_average_loss(batch_ave_loss, node_count)
            
            self.record_ave_training_loss.append(ave_loss)
        # convert to ms
        self.time_train_total = ((time.time() - t0) * 1000)

    def whole_cpu_validate(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.test_device = torch.device("cpu")
        test_model = self.model.to(self.test_device)
        test_model.eval()   # set into test mode, only effective for certain modules such as dropout and batchNorm
        
        batch_file_name = self.data_folder + 'validation/batch_whole'

        t2 = time.time()
        with open(batch_file_name, "rb") as fp:
            minibatch_data_validation = pickle.load(fp)
        read_time = (time.time() - t2) * 1000
        print('*** During validation for # {0} batch, reading batch file costed {1:.2f} ms ***'.format("whole graph", read_time) )

        valid_validation_nodes, valid_edges, valid_edge_weights, valid_features, valid_target = minibatch_data_validation

        prediction = test_model(valid_edges, valid_features, valid_edge_weights)
        # select the testing nodes predictions and real labels
        predictions = prediction[valid_validation_nodes].cpu().detach().numpy()
        targets = valid_target[valid_validation_nodes].cpu().detach().numpy()
        
        # along axis:    axis == 1
        predictions = predictions.argmax(1)  # return the indices of maximum probability 
        
        f1 = f1_score(targets, predictions, average="micro")
        accuracy = accuracy_score(targets, predictions)
#         print("\nTest F-1 score: {:.4f}".format(score))
        return (f1, accuracy)

    def middle_check_whole_cpu_validate(self, minibatch_data_validation):
        """
        Scoring the test and printing the F-1 score.
        """
        test_model = copy.deepcopy(self.model)
        test_model = test_model.to(self.test_device)
        test_model.eval()   # set into test mode, only effective for certain modules such as dropout and batchNorm
        

        valid_validation_nodes, valid_edges, valid_edge_weights, valid_features, valid_target = minibatch_data_validation

        prediction = test_model(valid_edges, valid_features, valid_edge_weights)
        # select the testing nodes predictions and real labels
        predictions = prediction[valid_validation_nodes].cpu().detach().numpy()
        targets = valid_target[valid_validation_nodes].cpu().detach().numpy()
        
        # along axis:    axis == 1
        predictions = predictions.argmax(1)  # return the indices of maximum probability 
        
        f1 = f1_score(targets, predictions, average="micro")
        accuracy = accuracy_score(targets, predictions)
#         print("\nTest F-1 score: {:.4f}".format(score))
        return (f1, accuracy)

