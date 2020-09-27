### inside train module
import torch
import time
import os
import pickle
import dill

from metric import *
from batch_machine import *
from model_graphsaint import GraphSAINT
from utils import *


# define a lambda func
f_mean = lambda l: sum(l)/len(l)

def evaluate_full_batch(model, minibatch, mode='val'):
    """
        Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes.
        mode: can be val or test
    """
    loss, preds, labels = model.eval_step(*minibatch.one_batch(mode = mode))
    node_val_test = minibatch.node_val if mode=='val' else minibatch.node_test
    # may not be necessary 
    f1_scores = calc_f1(to_numpy(labels[node_val_test]), to_numpy(preds[node_val_test]), model.sigmoid_loss)
    return loss, f1_scores[0], f1_scores[1]


def train_setting(dataname, datapath, train_config_file):
    """
        YAML (a recursive acronym for "YAML Ain't Markup Language") is a human-readable data-serialization language. 
        It is commonly used for configuration files and in applications where data is being stored or transmitted.
    
        yaml.load is as powerful as pickle.load and so may call any Python function. Check the yaml.safe_load function though.
        The function yaml.load converts a YAML document to a Python object.
    """
    with open(train_config_file) as f_train_config:
        train_config = yaml.load(f_train_config)
        
    arch_gcn = {'dim':-1,
                'aggr':'concat',
                'loss':'softmax',
                'arch':'1',
                'act':'I',
                'bias':'norm'}
    # check the loss:  default to be softmax, multi-class problem, each node can only belong to just one class at last
    arch_gcn.update(train_config['network'][0])   # train_config['network'] is a list of dict
    
    
    train_params = {'lr' : 0.01, 'weight_decay' : 0., 'norm_loss':True, 'norm_aggr':True, 'q_threshold' : 50, 'q_offset':0}
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data(dataname, datapath = datapath)
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")
    
    # train_data is a tuple: adj_full, adj_train, feats, class_arr, role
    return train_params, train_phases, train_data, arch_gcn

def prepare(working_dir, train_data, train_params, arch_gcn):
    """
        working_dir: main working dir for experiments
        train_params: contain settings for the mini-batch setting
        arch_gcn: contain all the settings 
    """
    adj_full, adj_train, feat_full, class_arr, role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]
    
    # key switch :  cpu_eval (bool)
    # establish two models, one for train, one for evaluation, because later the model_eval will load the trained model parameters
    
    # for training process: on GPU
    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    # for evaluation: validaiton/test  : on CPU
    minibatch_eval = Minibatch(adj_full_norm, adj_train,role, train_params, cpu_eval=True)
    model_eval = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    # model, model_eval, mini_batch_eval can be saved as pickle file for use later

    ### but cannot pickle lambda func for now

    prepare_data_folder = working_dir + 'prepare_data/'
    os.makedirs(os.path.dirname(prepare_data_folder), exist_ok=True)

    train_input_file_name = prepare_data_folder + 'model_train_input'
    with open(train_input_file_name, "wb") as fp:
        dill.dump((minibatch, model), fp)

    evaluation_input_file_name = prepare_data_folder + 'model_eval_input'
    with open(evaluation_input_file_name, "wb") as fp:
        dill.dump((minibatch_eval, model_eval), fp)
    
    # return model, minibatch, minibatch_eval, model_eval


def train_investigate(snap_model_folder, train_phases, model, minibatch, eval_train_every, snapshot_every = 10,
          mini_epoch_num = 5, multilabel = True, core_par_sampler = 1, samples_per_processor = 200):
    """
    PURPOSE:  to go through each training phase and take a snapshot of current mode and saved as pickle files
        snap_model_folder : folder to save the model snapshots during training
        train_phases:  use defined train fases defined in the .yml file
        model :  graphsaint model for training
        minibatch:   minibatch for training, usually with batches pool
        eval_train_every :  periodically store the train loss during the training process
        snapshot_every :  periodically store the states of the trained model for later evaluation
        mini_epoch_num :  how long the training will focus on one single batch
        multilabel : True if a multi-label task, otherwise a multi-class case
        core_par_sampler : how many CPU cores  will be used on each CPU
        samples_per_processor : how many samples will be generated from each CPU
    return: 1) total training time;  2) data uploading time
    
    """
    
    epoch_ph_start = 0
    time_train, time_upload, pure_time_train = 0, 0, 0
    
    # establish a new folder if it does not exist
#     os.makedirs(snap_model_folder, exist_ok = True)
    
    for ip, phase in enumerate(train_phases):
        printf('START PHASE {:4d}'.format(ip),style='underline')
        
        minibatch.set_sampler(phase, core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)  # one of the phases defined in the phase section of the .yml file, here phase is a dict
        num_batches = minibatch.num_training_batches()
#         print('calculated batch number is: ', num_batches)
        
        
        epoch_ph_end = int(phase['end'])
        macro_epoch_part_num = (epoch_ph_end - epoch_ph_start) // mini_epoch_num
        for macro_epoch_idx in range(macro_epoch_part_num):
            
            minibatch.shuffle()  # each time shuffle, restore the self.batch_num back to -1
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            
            actual_batch_num = 0
            
#             while not minibatch.end():
            for batch_idx in range(num_batches):
                
                node_subgraph, adj_subgraph, norm_loss_subgraph = minibatch.one_batch(mode='train')   # here minibatch should be an interator
                # redefine all the function interfaces to explicitely upload and run the training
                
                ### =========== prepare for all the data to be used ==============
                feat_subg = model.feat_full[node_subgraph]
                label_subg = model.label_full[node_subgraph]
                
                if not multilabel:
                    # for the multi-class, need type conversion
                    label_full_cat = torch.from_numpy(model.label_full.numpy().argmax(axis=1).astype(np.int64))

                label_subg_converted = label_subg if multilabel else label_full_cat[node_subgraph]
                
                t0 = time.time()
                # transfer data to the GPU
                feat_subg = feat_subg.cuda()
                label_subg = label_subg.cuda()
                adj_subgraph = adj_subgraph.cuda()
                norm_loss_subgraph = norm_loss_subgraph.cuda()
                label_subg_converted = label_subg_converted.cuda()
                time_upload += time.time() - t0
                
                
                for micro_epoch_idx in range(mini_epoch_num):
                    real_epoch_idx = 1 + micro_epoch_idx + macro_epoch_idx * mini_epoch_num + epoch_ph_start
                    printf('Epoch {:4d}, Batch ID {}'.format(real_epoch_idx, batch_idx),style='bold')
                    # pure training process:
                    t1 = time.time()
                    loss_train, preds_train = \
                            model.train_step(node_subgraph, adj_subgraph, norm_loss_subgraph, feat_subg, label_subg_converted)
                    
                    pure_time_train += time.time() - t1
                    
                    # take a snapshot of current model
                    if batch_idx == num_batches - 1 and real_epoch_idx % snapshot_every == 0:
                        snap_model_file = snap_model_folder + 'snapshot_epoch_' + str(real_epoch_idx) + '.pkl'
                        torch.save(model.state_dict(), snap_model_file)  # store the current state_dict() into the file: 'tmp.pkl'
                    
                    labels_train = label_subg
                    # periodically calculate all the statistics and store them
                    if not minibatch.batch_num % eval_train_every:
                        # to_numpy already convert tensor onto CPU
                        f1_mic, f1_mac = calc_f1(to_numpy(labels_train), to_numpy(preds_train), model.sigmoid_loss)
                        l_loss_tr.append(loss_train)
                        l_f1mic_tr.append(f1_mic)
                        l_f1mac_tr.append(f1_mac)
            
            
        # for different training phase, train it continuously 
        epoch_ph_start = int(phase['end'])
        printf("Optimization Finished!", style="yellow")
    
    time_train = pure_time_train + time_upload
    # after going through all the training phases, print out the total time
    printf("Total training time: {:6.2f} ms".format(time_train * 1000), style='red')
    printf("Total train data uploading time: {:6.2f} ms".format(time_upload * 1000), style='red')
    
    return time_train * 1000, time_upload * 1000
            

def evaluate(snap_model_folder, minibatch_eval, model_eval, epoch_idx, mode='val'):
    """
        Perform the evaluation: either validaiton or test offline from saved snapshot of the models
        generate evaluation results from a single timepoint snapshot of the trained model
        return : micro_f1 score, macro_f1 score
        
    """
    ### location to output the evaluation result
    
    snap_model_file = snap_model_folder + 'snapshot_epoch_' + str(epoch_idx) + '.pkl'

    model_eval.load_state_dict(torch.load(snap_model_file, map_location=lambda storage, loc: storage))

    loss_val, f1mic_val, f1mac_val = evaluate_full_batch(model_eval, minibatch_eval, mode = mode)

    return f1mic_val, f1mac_val
    


