
import time
import torch
import random
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


from Custom_GCNConv import Net

#####################################
###      Mini batch Trainer       ###
#####################################
class ClusterGCNTrainer_mini_Train(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, clustering_machine, in_channels, out_channels, input_layers = [32, 16], dropout=0.3):
        """
        :param in_channels, out_channels: input and output feature dimension
        :param clustering_machine:
        """  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clustering_machine = clustering_machine
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
    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
#         print('inside overlapping do forward for cluster : ', cluster)
        
        t1 = time.time()
        edges = self.clustering_machine.sg_mini_edges_local[cluster].to(self.device)
        features = self.clustering_machine.sg_mini_features[cluster].to(self.device)
        edge_weights = self.clustering_machine.sg_mini_edge_weight_local[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_mini_train_nodes_local[cluster].to(self.device)
        
        self.time_train_load_data += (time.time() - t1) * 1000
        
        # torch.squeeze()  removes all the dimension with value 1, change the target from 2-D  (N by 1) into 1-D N tensor
        target = self.clustering_machine.sg_mini_labels[cluster].to(self.device)
        '''Target and features are one-one mapping'''
        # calculate the probabilites from log_sofmax
        predictions = self.model(edges, features, edge_weights)
        
        ave_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
        node_count = train_nodes.shape[0]

        # for each cluster keep track of the counts of the nodes
        return ave_loss, node_count


    def update_average_loss(self, batch_average_loss, node_count, isolate = True):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        if isolate:
            self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss / self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        test_nodes = self.clustering_machine.sg_test_nodes_global[cluster].to(self.device)
        prediction = self.model(self.edges, self.features, self.edge_weights)
        
        return prediction[test_nodes], self.label[test_nodes]

    # iterate through epoch and also the clusters
    def train(self, epoch_num=10, learning_rate=0.01, weight_decay = 0.01):
        """
        Training a model.
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model.train()
        self.record_ave_training_loss = []
        self.time_train_load_data = 0
        
        t0 = time.time()
        for epoch in range(epoch_num):
#             For test purpose, we let the clusters to follow specific order
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
#             for cluster in [0]:
#                 print('current cluster is: ', cluster)
                self.optimizer.zero_grad()
                batch_ave_loss, node_count = self.do_forward_pass(cluster)
                batch_ave_loss.backward()
                self.optimizer.step()
                ave_loss = self.update_average_loss(batch_ave_loss, node_count)
            
            self.record_ave_training_loss.append(ave_loss)
        # convert to ms
        self.time_train_total = ((time.time() - t0) * 1000)
        
#         epochs.set_description("Ave Train Loss per node: %g " % round(ave_loss,6))
#         print("Train ave loss of overlapping clusters per node : %g" % round(ave_loss,6))

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        
        self.edges = self.clustering_machine.edge_index_global_self_loops.to(self.device)
        self.features = self.clustering_machine.features.to(self.device)
        self.edge_weights = self.clustering_machine.edge_weight_global.to(self.device)
        self.label = self.clustering_machine.label.to(self.device)
        
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)

            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        
        # concatenate all the ndarrays inside this list
        self.targets = np.concatenate(self.targets)
        # along axis:    axis == 1
        self.predictions = np.concatenate(self.predictions).argmax(1)  # return the indices of maximum probability 
#         print('shape of the targets and predictions are: ', self.targets.shape, self.predictions.shape)
        
        f1 = f1_score(self.targets, self.predictions, average="micro")
        accuracy = accuracy_score(self.targets, self.predictions)
#         print("\nTest F-1 score: {:.4f}".format(score))
        return (f1, accuracy)
    
    def do_validation_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        validation_nodes = self.clustering_machine.sg_validation_nodes_global[cluster].to(self.device)
        prediction = self.model(self.edges, self.features, self.edge_weights)
        
        return prediction[validation_nodes], self.label[validation_nodes]

    def validate(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        
        self.edges = self.clustering_machine.edge_index_global_self_loops.to(self.device)
        self.features = self.clustering_machine.features.to(self.device)
        self.edge_weights = self.clustering_machine.edge_weight_global.to(self.device)
        self.label = self.clustering_machine.label.to(self.device)
        
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_validation_prediction(cluster)

            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        
        # concatenate all the ndarrays inside this list
        self.targets = np.concatenate(self.targets)
        # along axis:    axis == 1
        self.predictions = np.concatenate(self.predictions).argmax(1)  # return the indices of maximum probability 
#         print('shape of the targets and predictions are: ', self.targets.shape, self.predictions.shape)
        
        f1 = f1_score(self.targets, self.predictions, average="micro")
        accuracy = accuracy_score(self.targets, self.predictions)
#         print("\nTest F-1 score: {:.4f}".format(score))
        return (f1, accuracy)

#####################################
###      Whole Input Trainer      ###
#####################################
class wholeClusterGCNTrainer_sequence(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, clustering_machine, in_channels, out_channels, input_layers = [16, 16], dropout=0.3):
        """
        :param in_channels, out_channels: input and output feature dimension
        :param clustering_machine:
        """  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clustering_machine = clustering_machine
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
    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        
        train_nodes = self.clustering_machine.sg_train_nodes_global[cluster].to(self.device)
        
        '''Target and features are one-one mapping'''
        # calculate the probabilites from log_sofmax
        predictions = self.model(self.edges, self.features, self.edge_weights)
        
        ave_loss = torch.nn.functional.nll_loss(predictions[train_nodes], self.label[train_nodes])
        node_count = train_nodes.shape[0]

        # for each cluster keep track of the counts of the nodes
        return ave_loss, node_count


    def update_average_loss(self, batch_average_loss, node_count, isolate = True):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        if isolate:
            self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss / self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        test_nodes = self.clustering_machine.sg_test_nodes_global[cluster].to(self.device)
        target = self.clustering_machine.label.to(self.device)
        prediction = self.model(self.edges, self.features, self.edge_weights)
        
        return prediction[test_nodes], target[test_nodes]

    # iterate through epoch and also the clusters
    def train(self, epoch_num=10, learning_rate=0.01, weight_decay = 0.01):
        """
        Training a model.
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model.train()   # set self.training = True : will be used in the dropout
        self.record_ave_training_loss = []
        self.time_train_load_data = 0
        
        t0 = time.time()
        # first transfer the whole graph data to the GPU device
        
        t1 = time.time()
        self.edges = self.clustering_machine.edge_index_global_self_loops.to(self.device)
        self.features = self.clustering_machine.features.to(self.device)
        self.edge_weights = self.clustering_machine.edge_weight_global.to(self.device)
        self.label = self.clustering_machine.label.to(self.device)
        self.time_train_load_data += (time.time() - t1) * 1000
        
        for epoch in range(epoch_num):
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_ave_loss, node_count = self.do_forward_pass(cluster)
                batch_ave_loss.backward()
                self.optimizer.step()
                ave_loss = self.update_average_loss(batch_ave_loss, node_count)
            
            self.record_ave_training_loss.append(ave_loss)
        # convert to ms
        self.time_train_total = ((time.time() - t0) * 1000)
        
#         epochs.set_description("Ave Train Loss per node: %g " % round(ave_loss,6))
#         print("Train ave loss of overlapping clusters per node : %g" % round(ave_loss,6))

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()   # set self.training = false
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)

            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        
        # concatenate all the ndarrays inside this list
        self.targets = np.concatenate(self.targets)
        # along axis:    axis == 1
        self.predictions = np.concatenate(self.predictions).argmax(1)  # return the indices of maximum probability 
#         print('shape of the targets and predictions are: ', self.targets.shape, self.predictions.shape)
        
        f1_score = f1_score(self.targets, self.predictions, average="micro")
        accuracy = accuracy_score(self.targets, self.predictions)
#         print("\nTest F-1 score: {:.4f}".format(score))
        return (f1_score, accuracy)
    
    def do_validation_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        validation_nodes = self.clustering_machine.sg_validation_nodes_global[cluster].to(self.device)
        target = self.clustering_machine.label.to(self.device)
        
        prediction = self.model(self.edges, self.features, self.edge_weights)
        
        return prediction[validation_nodes], target[validation_nodes]

    def validate(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_validation_prediction(cluster)

            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        
        # concatenate all the ndarrays inside this list
        self.targets = np.concatenate(self.targets)
        # along axis:    axis == 1
        self.predictions = np.concatenate(self.predictions).argmax(1)  # return the indices of maximum probability 
#         print('shape of the targets and predictions are: ', self.targets.shape, self.predictions.shape)
        
        f1 = f1_score(self.targets, self.predictions, average="micro")
        accuracy = accuracy_score(self.targets, self.predictions)
#         print("\nTest F-1 score: {:.4f}".format(score))
        return (f1, accuracy)

