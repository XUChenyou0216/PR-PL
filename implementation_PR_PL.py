# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:02:40 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:20:11 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:54:21 2021

@author: user
"""

import numpy as np

import torch
#import ctypes
#ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import random
import matplotlib.pyplot as plt
from torch.optim import Adam,SGD,RMSprop
from typing import Optional
import scipy.io as scio
from torch.optim.optimizer import Optimizer
from sklearn import preprocessing
from Adversarial import DomainAdversarialLoss
from model_PR_PL import Domain_adaption_model,discriminator
def setup_seed(seed): ## setup the random seed 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class StepwiseLR_GRL: 
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75,max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter
    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1

def weigth_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()    
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
#        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()


def one_hot(labels, num_classes):
    labels = labels.reshape(-1).astype('int64')
    encoded = np.zeros((len(labels), num_classes), dtype='float32')
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


DATASET_CACHE = {}


def load_session_dataset(session):
    if session in DATASET_CACHE:
        return DATASET_CACHE[session]

    data_path = 'D:/EGG_dataset/SEED-IV/feature_seediv'
    dataset_key = 'dataset_session{}'.format(session)
    num_subjects = 15
    num_classes = 4
    feature_list = []
    label_list = []

    for subj_id in range(1, num_subjects + 1):
        file_name = 'sub_{}_session_{}.mat'.format(subj_id, session)
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError('Cannot find SEED-IV feature file: {}'.format(file_path))

        mat = scio.loadmat(file_path)
        feature = mat[dataset_key]['feature'][0, 0].astype('float32')
        label = mat[dataset_key]['label'][0, 0].astype('int64') - 1

        feature_list.append(feature)
        label_list.append(one_hot(label, num_classes))

    DATASET_CACHE[session] = (feature_list, label_list)
    return DATASET_CACHE[session]


def get_dataset(test_id, session):
    feature_list, label_list = load_session_dataset(session)
    source_features = feature_list[:test_id] + feature_list[test_id + 1:]
    source_labels = label_list[:test_id] + label_list[test_id + 1:]

    target_set = {'feature': feature_list[test_id], 'label': label_list[test_id]}
    source_set = {'feature': np.vstack(source_features), 'label': np.vstack(source_labels)}
    return target_set, source_set

def get_generated_targets(model,x_s,x_t,labels_s): ## Get generated labels by threshold
        with torch.no_grad():
            model.eval()
            _,_,_,_,dist_matrix = model(x_s,x_t,labels_s)     
            sim_matrix = model.get_cos_similarity_distance(labels_s)
            sim_matrix_target = model.get_cos_similarity_by_threshold(dist_matrix)
            return sim_matrix,sim_matrix_target
        
def checkpoint(model,checkpoint_PATH,flag):## saving or loading the checkpoint model
    if flag=='load':
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        model.P=model_CKPT['P']
        model.stored_mat=model_CKPT['stored_mat']
        model.cluster_label=model_CKPT['cluster_label']
        model.upper_threshold=model_CKPT['upper_threshold']
        model.lower_threshold=model_CKPT['lower_threshold']
        model.threshold=model_CKPT['threshold']
    elif flag=='save':
        torch.save({'P': model.P, 'stored_mat':model.stored_mat,'cluster_label':model.cluster_label,'threshold':model.threshold,
                    'upper_threshold':model.upper_threshold,'lower_threshold':model.lower_threshold,'state_dict': model.state_dict()},checkpoint_PATH)
    
def infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def train_model(loader_train, loader_test,model,dann_loss, optimizer,scheduler,hidden_4,epoch,batch_size,parameter,threshold_update=True): ## model training 
    # switch to train mode
    model.train()
    dann_loss.train()
    train_source_iter = infinite_iter(loader_train)
    train_target_iter = infinite_iter(loader_test)
    # train source and target with repeated target batches, matching the best previous SEED-IV baseline
    T = len(loader_train.dataset)//batch_size
    cls_loss_sum=0
    transfer_loss_sum=0
    if parameter['boost_type']=='linear':
        boost_factor=parameter['cluster_weight']*(epoch/model.max_iter)
    elif parameter['boost_type']=='exp':
        boost_factor=parameter['cluster_weight']*(2.0 / (1.0 + np.exp(-1 * epoch / model.max_iter))- 1)
    elif parameter['boost_type']=='constant':
        boost_factor=parameter['cluster_weight']
    for i in range(T):  
        # loading data: x_s:source feature,x_t:target feature,labels_s:source label
        model.train()
        x_s, labels_s = next(train_source_iter)
        x_s, labels_s = Variable(x_s.cuda()), Variable(labels_s.cuda()) 
        x_t, _ = next(train_target_iter)
        x_t=Variable(x_t.cuda())
        estimated_sim_truth,estimated_sim_truth_target = get_generated_targets(model,x_s,x_t,labels_s)
        # estimated_sim_truth: estimated pairwise matrix for source features(batch size * batch size)
        # estimated_sim_truth_target: estimated pairwise matrix for target features(batch size * batch size)
        _,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target = model(x_s,x_t,labels_s)
        eta=0.00001
        # pairwise loss matrix on source domain
        bce_loss=-(torch.log(sim_matrix+eta)*estimated_sim_truth)-(1-estimated_sim_truth)*torch.log(1-sim_matrix+eta)
        # pairwise loss matrix on target domain
        bce_loss_target=-(torch.log(sim_matrix_target+eta)*estimated_sim_truth_target)-(1-estimated_sim_truth_target)*torch.log(1-sim_matrix_target+eta)
        cls_loss = torch.mean(bce_loss)
        # valid pair selection for the target domain
        indicator,nb_selected=model.compute_indicator(sim_matrix_target)
        cluster_loss=torch.sum(indicator*bce_loss_target)/nb_selected
        # regularization  
        P_loss=torch.norm(torch.matmul(model.P.T,model.P)-torch.eye(hidden_4).cuda(),'fro')
        # domain adversarial loss
        transfer_loss = dann_loss(feature_source_f+0.005*torch.randn((batch_size,hidden_4)).cuda(),feature_target_f+0.005*torch.randn((batch_size,hidden_4)).cuda())
    
        cls_loss_sum+=cls_loss.data
        transfer_loss_sum+=transfer_loss.data
        loss =cls_loss+transfer_loss+0.01*P_loss+boost_factor*cluster_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('transfer_loss:',str(transfer_loss_sum/T))
    print('cls_loss:',str(cls_loss_sum/T))
    if threshold_update==True:
        model.update_threshold(epoch)  
    return cls_loss_sum.cpu().detach().numpy(),transfer_loss_sum.cpu().detach().numpy()

def train_and_test_GAN(test_id,max_iter,parameter,session,threshold_update=True): ## pipeline for PR-PL model
    setup_seed(20)
    hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,upper_threshold,lower_threshold=parameter['hidden_1'],parameter['hidden_2'],parameter['hidden_1'],parameter['hidden_2'],parameter['num_of_class'],parameter['low_rank'],parameter['upper_threshold'],parameter['lower_threshold']     
    BATCH_SIZE = 96
    ## dataloader(test_id: test subject in the LOOCV process, session:1,2,3 for different sessions in the SEED dataset)
    target_set,source_set=get_dataset(test_id,session)
    torch_dataset_train = Data.TensorDataset(torch.from_numpy(source_set['feature']),torch.from_numpy(source_set['label']))
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature']),torch.from_numpy(target_set['label']))
    test_features,test_labels=torch.from_numpy(target_set['feature']),torch.from_numpy(target_set['label'])
    source_features,source_labels=torch.from_numpy(source_set['feature']),torch.from_numpy(source_set['label'])
    test_features_cuda = test_features.cuda()
    test_labels_cuda = test_labels.cuda()
    source_features_cuda = source_features.cuda()
    source_labels_cuda = source_labels.cuda()
    loader_train = Data.DataLoader(
            dataset=torch_dataset_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True
            )
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True
            )
    setup_seed(20)
    # model initialization
    model=Domain_adaption_model(hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold).cuda(0)
    model.apply(weigth_init)
    domain_discriminator = discriminator(hidden_2).cuda()
    domain_discriminator.apply(weigth_init)
    dann_loss = DomainAdversarialLoss(domain_discriminator).cuda()
    optimizer = RMSprop(model.get_parameters() + domain_discriminator.get_parameters(),lr=1e-3, weight_decay=1e-5)
    lr_scheduler = None
    best_acc = 0. 
    target_acc_list=np.zeros(max_iter)
    target_nmi_list=np.zeros(max_iter)
    source_acc_list=np.zeros(max_iter)
    source_nmi_list=np.zeros(max_iter)
    cls_loss_list=np.zeros(max_iter)
    transfer_loss_list=np.zeros(max_iter)
    ## model training and evaluation
    for epoch in range(max_iter):
        # train for one epoch
        model.train()
        cls_loss_sum,transfer_loss_sum=train_model(loader_train, loader_test,model,dann_loss, optimizer,lr_scheduler,hidden_4,epoch,BATCH_SIZE,parameter,threshold_update)
        source_acc,source_nmi=model.cluster_label_update(source_features_cuda,source_labels_cuda)
        # evaluate on target domain
        model.eval()
        target_acc,target_nmi=model.target_domain_evaluation(test_features_cuda,test_labels_cuda)
        target_acc_list[epoch]=target_acc
        source_acc_list[epoch]=source_acc
        target_nmi_list[epoch]=target_nmi
        source_nmi_list[epoch]=source_nmi
        cls_loss_list[epoch]=cls_loss_sum
        transfer_loss_list[epoch]=transfer_loss_sum
        print('Test Subject:',test_id,'Session:',session)
        print('src:','epoch:',epoch,'acc=',source_acc,'nmi=',source_nmi)
        print('tar:','epoch:',epoch,'acc=',target_acc,'nmi=',target_nmi)
        best_acc = max(target_acc, best_acc)
    return best_acc,cls_loss_list,source_acc_list,source_nmi_list,target_acc_list,target_nmi_list,transfer_loss_list

def main(update_threshold,parameter,session):
    # update_threshold: if update_threshold == True,then update the dynamic sample selection threshold iteratively
    setup_seed(20)
    max_iter=1000
    best_acc_mat=np.zeros(15)
    transfer_loss_curve=np.zeros((15,max_iter))
    cls_loss_curve=np.zeros((15,max_iter))
    source_acc_curve=np.zeros((15,max_iter))
    target_acc_curve=np.zeros((15,max_iter))
    source_nmi_curve=np.zeros((15,max_iter))
    target_nmi_curve=np.zeros((15,max_iter))
    for i in range(15):
        best_acc,cls_loss_list,source_acc_list,source_nmi_list,target_acc_list,target_nmi_list,transfer_loss_list=train_and_test_GAN(i,max_iter,parameter,session,update_threshold)
        best_acc_mat[i]=best_acc
        source_acc_curve[i,:]=source_acc_list
        target_acc_curve[i,:]=target_acc_list
        source_nmi_curve[i,:]=source_nmi_list
        target_nmi_curve[i,:]=target_nmi_list
        transfer_loss_curve[i,:]=transfer_loss_list
        cls_loss_curve[i,:]=cls_loss_list
    return best_acc_mat,cls_loss_curve,transfer_loss_curve,source_acc_curve,source_nmi_curve,target_acc_curve,target_nmi_curve

## hidden_1: the number of input neurons of the hidden layer 1 (feature extractor)
## hidden_2: the number of input neurons of the hidden layer 2 (feature extractor)
## num_of_class: the number of classes
## cluster_weight: the coefficient of the unsupervised pairwise learning
## low_rank:  the rank of the low-rank matrix which is used for the Low-Rank-Approximation of the bilinear-matrix S
## upper_threshold: The initial upper_threshold for the non-linear dynamic threshold
## lower_threshold: The initial lower_threshold for the non-linear dynamic threshold
## boost_type:linear dynamic strategy for the unsupervised pairwise learning
## hyperparameter setting for the SEED-IV dataset    
parameter={'hidden_1':64,'hidden_2':64,'num_of_class':4,'cluster_weight':2,'low_rank':32,'upper_threshold':0.9,'lower_threshold':0.5,'boost_type':'linear'}
## hyperparameter setting for the SEED dataset    
#parameter={'hidden_1':64,'hidden_2':64,'num_of_class':3,'cluster_weight':2,'low_rank':32,'upper_threshold':0.9,'lower_threshold':0.5,'boost_type':'linear'}
def run_session(session):
    best_acc_mat,cls_loss_curve,transfer_loss_curve,source_acc_curve,source_nmi_curve,target_acc_curve,target_nmi_curve=main(True,parameter,session)
    result_list={'best_acc_mat':best_acc_mat,
                 'cls_loss_curve':cls_loss_curve,
                 'source_acc_curve':source_acc_curve,
                 'source_nmi_curve':source_nmi_curve,
                 'target_acc_curve':target_acc_curve,
                 'target_nmi_curve':target_nmi_curve}

    print(f"\n========== SEED-IV Session {session} Final Result ==========")
    print("Best accuracy for each subject:")
    for i, acc in enumerate(best_acc_mat):
        print(f"  Subject {i+1:2d}: {acc*100:.2f}%")

    print(f"\nMean accuracy: {np.mean(best_acc_mat)*100:.2f}%")
    print(f"Std:           {np.std(best_acc_mat)*100:.2f}%")
    print(f"Final result:  {np.mean(best_acc_mat)*100:.2f}% +/- {np.std(best_acc_mat)*100:.2f}%")
    print(f"\nPaper result:  81.32% +/- 8.53%")
    print(f"Gap:           {abs(np.mean(best_acc_mat)*100 - 81.32):.2f}%")

    scio.savemat(f'result_SEED_IV_session_{session}.mat', result_list)


if __name__ == '__main__':
    for session in [2, 3]:
        run_session(session)
