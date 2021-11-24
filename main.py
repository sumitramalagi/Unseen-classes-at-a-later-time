import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import scipy.io
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import argparse
import os
import pdb
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#### Specify the appropriate dataloader here ### 
from Dataloader_dynamic import DATA_LOADER as dataloader
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity

cwd=os.path.dirname(os.getcwd())

torch.manual_seed(55)
torch.cuda.manual_seed(55)
    
class Generator(nn.Module):
    def __init__(self, feature_size=2048, att_size=85):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2*att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, noise,att):
        if len(att.shape)==3:
            h = torch.cat((noise, att), 2) 
        else:
            h = torch.cat((noise, att), 1) 
        feature = torch.relu(self.fc1(h)) 
        feature = torch.sigmoid(self.fc2(feature))
        return feature
    
class Discriminator(nn.Module):  
    def __init__(self, feature_size=2048, att_size=85):
        super(Discriminator, self).__init__()         
        self.fc1 = nn.Linear(att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    def forward(self, att): 
        att_embed = torch.relu(self.fc1(att))
        att_embed = torch.relu(self.fc2(att_embed))
        return att_embed
 
def compute_D_acc(discriminator,test_dataloader,seen_classes,novel_classes,task_no,batch_size=128, opt1='gzsl', opt2='test_seen',psuedo_ft = None, psuedo_lb = None,trial=0):
    if psuedo_ft is not None:
        data = Data.TensorDataset(psuedo_ft, psuedo_lb)
        test_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)   
    else:
        test_loader = test_dataloader.get_loader(opt2, batch_size=128)
    att = test_dataloader.data['whole_attributes'].cuda()
    if opt1 == 'gzsl':    
        search_space = np.arange(att.shape[0])
    if opt1 == 'zsl':
        search_space = test_dataloader.data['unseen_label']

    pred_label = []
    true_label = []
    with torch.no_grad():
        for features, labels in test_loader:    
            features, labels = features.cuda(), labels.cuda()
            features = F.normalize(features, p=2, dim=-1, eps=1e-12)  
            if psuedo_ft is None:
                 features = features.unsqueeze(1).repeat(1, search_space.shape[0], 1)
            else:
                features = features.squeeze(1).unsqueeze(1).repeat(1, search_space.shape[0], 1)
            semantic_embeddings = discriminator(att).cuda() 
            semantic_embeddings= F.normalize(semantic_embeddings, p=2, dim=-1, eps=1e-12) 
            cosine_sim = F.cosine_similarity(semantic_embeddings, features, dim=-1) 
            predicted_label = torch.argmax(cosine_sim, dim=1)
            predicted_label = search_space[predicted_label.cpu()]
            pred_label = np.append(pred_label, predicted_label)
            true_label = np.append(true_label, labels.cpu().numpy())
    pred_label = np.array(pred_label, dtype='int')
    true_label = np.array(true_label, dtype='int')
    acc = 0
    unique_label = np.unique(true_label)
    for i in unique_label:
        idx = np.nonzero(true_label == i)[0]
        acc += accuracy_score(true_label[idx], pred_label[idx])
    acc = acc / unique_label.shape[0]
    return acc


class Train_Dataloader:
    def __init__(self,train_feat_seen,train_label_seen,batch_size=32):
        self.data = {'train_': train_feat_seen, 'train_label': train_label_seen} 
    def get_loader(self, opt='train_', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt+'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)        
        return data_loader

class Test_Dataloader:
    def __init__(self,test_attr,test_seen_f,test_seen_l,test_seen_a,test_unseen_f,test_unseen_l,test_unseen_a, batch_size=32):
        labels = torch.cat((test_seen_l,test_unseen_l))
        self.data = { 'test_seen': test_seen_f, 'test_seenlabel': test_seen_l,
        'whole_attributes': test_attr,
        'test_unseen': test_unseen_f, 'test_unseenlabel': test_unseen_l,
        'seen_label': np.unique(test_seen_l), 
        'unseen_label': np.unique(test_unseen_l)} 
    
    def get_loader(self, opt='test_seen', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt+'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)        
        return data_loader

def next_batch_unseen(unseen_attr, unseen_labels, batch_size):
    idx = torch.randperm(unseen_attr.shape[0])[0:batch_size]
    unsn_at = unseen_attr[idx]
    unsn_lbl = unseen_labels[idx]
    return unsn_at.unsqueeze(1), unsn_lbl

def next_batch(batch_size,attributes,feature,label):      
        idx = torch.randperm(feature.shape[0])[0:batch_size]
        batch_feature = feature[idx]
        batch_label = label[idx]
        batch_attr = attributes[idx]
        return batch_feature, batch_label, batch_attr

def train(task_no,Neighbors,discriminator,generator,data,seen_classes,novel_classes,replay_feat,replay_lab,replay_attr,feature_size=2048,attribute_size=85,no_of_replay=300,dlr=0.005,glr=0.005,batch_size=64,unsn_batch_size = 8,epochs=50,lambda1=10.0,alpha=1.0,beta=1.0,avg_feature=None,all_classes=None,num_tasks=None):    
    if task_no==1:
        train_feat_seen,train_label_seen,train_att_seen = data.task_train_data(task_no,seen_classes,all_classes,novel_classes,num_tasks,Neighbors)
        test_unseen_l,test_unseen_a = data.task_test_data(task_no,seen_classes,all_classes,novel_classes,num_tasks)
        avg_feature = torch.zeros((seen_classes), feature_size).float()
        cls_num = torch.zeros(seen_classes).float()
        
        for i,l1 in enumerate(train_label_seen):
            avg_feature[l1] += train_feat_seen[i]  
            cls_num[l1] += 1          

        for ul in np.unique(train_label_seen):
            avg_feature[ul] = avg_feature[ul]/cls_num[ul]

        avg_feature = avg_feature.cuda()
        semantic_relation_sn = data.idx_mat
        semantic_relation_unsn = data.unseen_idx_mat
        semantic_values_sn = data.semantic_similarity_seen
        semantic_values_unsn = data.semantic_similarity_unseen  
    
    else:  
        train_feat_seen,train_label_seen,train_att_seen = data.task_train_data(task_no,seen_classes,all_classes,novel_classes,num_tasks,Neighbors)
        test_unseen_l,test_unseen_a = data.task_test_data(task_no,seen_classes,all_classes,novel_classes,num_tasks)
        avg_feature_prev = avg_feature
        avg_feature = torch.zeros((seen_classes)*task_no, feature_size).float()
        cls_num = torch.zeros(seen_classes*task_no).float()
        avg_feature[:seen_classes*(task_no-1),:] = avg_feature_prev.cpu()

        for i,l1 in enumerate(train_label_seen):
            avg_feature[l1] += train_feat_seen[i]  
            cls_num[l1] += 1            

        for ul in np.unique(train_label_seen):
            avg_feature[ul] = avg_feature[ul]/cls_num[ul]
        
        avg_feature = avg_feature.cuda()
        semantic_relation_sn = data.idx_mat
        semantic_relation_unsn = data.unseen_idx_mat
        semantic_values_sn = data.semantic_similarity_seen
        semantic_values_unsn = data.semantic_similarity_unseen

    if task_no == 1:
        unseen_attr = test_unseen_a[int(task_no)]
        unseen_label = torch.reshape(test_unseen_l[int(task_no)],(test_unseen_l[int(task_no)].shape[0],1))

        whole_feat_seen = train_feat_seen
        whole_labels_seen = train_label_seen 
        whole_attr_seen = train_att_seen       
        replay = False

    if task_no>1:  
        for i in range(1,task_no+1):      
            if i==1:
                unseen_attr = test_unseen_a[int(i)]
                unseen_label = test_unseen_l[int(i)]
            else:
                unseen_attr = torch.cat((unseen_attr,test_unseen_a[int(i)]))
                unseen_label = torch.cat((unseen_label,test_unseen_l[int(i)]))
        unseen_label = unseen_label.unsqueeze(1)
        whole_feat_seen = torch.cat((train_feat_seen,replay_feat))
        whole_labels_seen = torch.cat((train_label_seen,replay_lab))
        whole_attr_seen = torch.cat((train_att_seen,replay_attr))
        replay = True
        

    train_loader = Train_Dataloader(whole_feat_seen,whole_labels_seen)
    train_ = whole_feat_seen.cuda() 
    whole_attr_seen = whole_attr_seen.cuda()  
    att_per_task = data.attribute_mapping(seen_classes,novel_classes,task_no).cuda()
    attr_seen_exc = att_per_task[0:seen_classes*task_no,:]
    _att_sn = attr_seen_exc.unsqueeze(0).repeat([batch_size,1,1])
    _att_unsn = att_per_task.unsqueeze(0).repeat([unsn_batch_size,1,1])
    train_label = whole_labels_seen 
    seen_label = np.unique(whole_labels_seen) 
    train_data_loader = train_loader.get_loader('train_', batch_size)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=dlr, weight_decay=0.00001)
    G_optimizer = optim.Adam(generator.parameters(), lr=glr, weight_decay=0.00001)
    entory_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_seen = 0
    best_unseen = 0
    best_H = 0
    A_D_seen = {}
    A_D_unseen = {}
    A_D_H = {}


    for epoch in range(epochs):
        print("Epoch {}/{}...".format(epoch + 1, epochs))
        for feature, label in train_data_loader:
            feature, label = feature.cuda(), label.cuda()   
            feature_norm = F.normalize(feature, p=2, dim=-1, eps=1e-12)  
            D_optimizer.zero_grad()
            att_bs_seen = att_per_task[label] 
            noise = torch.FloatTensor(att_bs_seen.shape[0], att_bs_seen.shape[1], att_bs_seen.shape[2]).cuda()  
            noise.normal_(0, 1)
            psuedo_seen_features = generator(noise,att_bs_seen) 
        
            semantic_embedding = discriminator(att_bs_seen)   
            semantic_embed_norm =  F.normalize(semantic_embedding, p=2, dim=-1, eps=1e-12)
            psuedo_seen_features_norm = F.normalize(psuedo_seen_features, p=2, dim=-1, eps=1e-12)
            real_cosine_similarity = lambda1 * F.cosine_similarity(semantic_embed_norm, feature_norm, dim=-1)  
            pseudo_cosine_similarity = lambda1 * F.cosine_similarity(semantic_embed_norm, psuedo_seen_features_norm, dim=-1)  
            real_cosine_similarity = torch.mean(real_cosine_similarity) 
            pseudo_cosine_similarity = torch.mean(pseudo_cosine_similarity)

            att_task_emb = discriminator(att_per_task[:(seen_classes)*task_no])
            mse_d = mse_loss(avg_feature,att_task_emb)
            _att_D = discriminator(_att_sn)  
            _att_D_norm = F.normalize(_att_D, p=2, dim=-1, eps=1e-12)
            real_features = feature_norm.unsqueeze(1).repeat([1, (seen_classes)*task_no, 1])
            real_cosine_sim = lambda1 * F.cosine_similarity(_att_D_norm, real_features, dim=-1)  
            cls_label = label
            classification_losses = entory_loss(real_cosine_sim, cls_label.squeeze()) 
            d_loss =  - torch.log(real_cosine_similarity) + torch.log(pseudo_cosine_similarity) + alpha * classification_losses+ mse_d
               
            d_loss.backward(retain_graph=True)
            D_optimizer.step()
            G_optimizer.zero_grad()
            fake_features = psuedo_seen_features_norm.repeat([1, seen_classes*task_no, 1]) 
            fake_cosine_sim = lambda1 * F.cosine_similarity(_att_D_norm, fake_features, dim=-1)
            pseudo_classification_loss = entory_loss(fake_cosine_sim, cls_label.squeeze())

            Euclidean_loss = Variable(torch.Tensor([0.0]), requires_grad=True).cuda()
            Correlation_loss = Variable(torch.Tensor([0.0]), requires_grad=True).cuda()  
            seen_sample_labels = np.unique(whole_labels_seen)
            
            for i in range(seen_classes*task_no): 
                sample_idx = (label == i)
                if (sample_idx==1).sum().item() == 0:
                    Euclidean_loss += 0.0
                if (sample_idx==1).sum().item() != 0:
                    G_sample_cls = psuedo_seen_features[sample_idx, :]
                    if G_sample_cls.shape[0] > 1:
                        generated_mean = G_sample_cls.mean(dim=0)
                    else:
                        generated_mean = G_sample_cls                                               
                    Euclidean_loss += (generated_mean - avg_feature[i]).pow(2).sum().sqrt()
                    for n in range(Neighbors):   
                        generated_mean_norm = F.normalize(generated_mean, p=2, dim=-1, eps=1e-12)  
                        avg_norm = F.normalize(avg_feature[semantic_relation_sn[i,n]], p=2, dim=-1, eps=1e-12)                                                
                        Neighbor_correlation = cosine_similarity(generated_mean_norm.data.cpu().numpy().reshape((1, 2048)),avg_norm.data.cpu().numpy().reshape((1, 2048)))
                        lower_limit = semantic_values_sn [i,n] - 0.01
                        if opt.dataset == "CUB":
                            upper_limit = semantic_values_sn [i,n] + 0.04
                        else:
                            upper_limit = semantic_values_sn [i,n] + 0.01
                        lower_limit = torch.as_tensor(lower_limit.astype('float')) 
                        upper_limit = torch.as_tensor(upper_limit.astype('float')) 
                        corr = torch.as_tensor(Neighbor_correlation[0][0].astype('float'))
                        margin = (torch.max(corr - corr, corr - upper_limit))**2 + (torch.max(corr- corr, lower_limit - corr ))**2 
                        Correlation_loss += margin           
            
            Euclidean_loss *= 1.0/(seen_classes*task_no) * 1
            lsr_seen = (Correlation_loss/5) * 20
            (5*lsr_seen).backward()            

            g_loss_seen = - torch.log(pseudo_cosine_similarity) + alpha * pseudo_classification_loss + Euclidean_loss 
            g_loss_seen.backward(retain_graph=True)
            G_optimizer.step()
          
            unseen_att, unseen_labl = next_batch_unseen(unseen_attr, unseen_label,unsn_batch_size)
            noise = torch.FloatTensor(unseen_att.shape[0], unseen_att.shape[1], unseen_att.shape[2]).cuda()   
            noise.normal_(0, 1)
            pseudo_unseen_features = generator(noise,unseen_att.cuda()) 

            unseen_cls_feature = pseudo_unseen_features.repeat([1, _att_unsn.shape[1], 1]) 
            _att_D_un = discriminator(_att_unsn)
            _att_D_un_norm = F.normalize(_att_D_un, p=2, dim=-1, eps=1e-12)
            unseen_cls_feature_norm = F.normalize(unseen_cls_feature, p=2, dim=-1, eps=1e-12)    
            pseudo_cos_sim_unsn = lambda1 * F.cosine_similarity(_att_D_un_norm, unseen_cls_feature_norm, dim=-1)   
            seen_normalized_ce_loss = entory_loss(pseudo_cos_sim_unsn, unseen_labl.cuda().squeeze())
    
            Correlation_loss_zero = Variable(torch.Tensor([0.0]), requires_grad= True).cuda() 
            unseen_sample_labels = np.unique(unseen_label) 

            for i in range(novel_classes*task_no):   
                    sample_idx = (unseen_labl == unseen_sample_labels[i])
                    if (sample_idx==1).sum().item() != 0:
                        G_sample_cls_zero = pseudo_unseen_features[sample_idx.reshape(unsn_batch_size)]
                        if G_sample_cls_zero.shape[0] > 1:
                            generated_mean = G_sample_cls_zero.mean(dim=0) 
                        else:
                            generated_mean = G_sample_cls_zero             
                        for n in range(Neighbors):   
                            generated_mean_norm = F.normalize(generated_mean, p=2, dim=-1, eps=1e-12)
                            avg_norm_un =  F.normalize(avg_feature[semantic_relation_unsn[i,n]], p=2, dim=-1, eps=1e-12)                                       
                            Neighbor_correlation = cosine_similarity(generated_mean_norm.data.cpu().numpy().reshape((1, 2048)), 
                                                    avg_norm_un.data.cpu().numpy().reshape((1, 2048)))                           
                            lower_limit = semantic_values_unsn [i,n] - 0.01                   
                            if opt.dataset == "CUB":
                                upper_limit = semantic_values_unsn [i,n] + 0.04
                            else:
                                upper_limit = semantic_values_unsn [i,n] + 0.01
                            lower_limit = torch.as_tensor(lower_limit.astype('float')) 
                            upper_limit = torch.as_tensor(upper_limit.astype('float')) 
                            corr = torch.as_tensor(Neighbor_correlation[0][0].astype('float'))
                            margin = (torch.max(corr- corr, corr - upper_limit))**2 + (torch.max(corr- corr, lower_limit - corr ))**2 
                            Correlation_loss_zero += margin    

            lsr_unsn = (Correlation_loss_zero/5) * 20
            g_loss_unseen =  alpha * (seen_normalized_ce_loss) + (5*lsr_unsn)      
            g_loss_unseen.backward(retain_graph=True)
            G_optimizer.step()

        for tno in range(1,task_no+1):
            test_seen_f,test_seen_l,test_seen_a,test_unseen_f,test_unseen_l,test_unseen_a = data.task_test_data_(tno,seen_classes,all_classes,novel_classes,num_tasks)
            att_per_task_ = data.attribute_mapping(seen_classes,novel_classes,tno).cuda()
            test_dataloader = Test_Dataloader(att_per_task_,test_seen_f,test_seen_l,test_seen_a,test_unseen_f,test_unseen_l,test_unseen_a)
            D_seen_acc = compute_D_acc(discriminator, test_dataloader,seen_classes,novel_classes,tno, batch_size = batch_size, opt1='gzsl', opt2='test_seen')
            D_unseen_acc = compute_D_acc(discriminator, test_dataloader,seen_classes,novel_classes,tno, batch_size = batch_size, opt1='gzsl', opt2='test_unseen')
            if D_unseen_acc==0 or D_seen_acc==0:
                D_harmonic_mean = 0
            else:
                D_harmonic_mean = (2*D_seen_acc*D_unseen_acc)/(D_seen_acc+D_unseen_acc)
            A_D_seen[tno] = D_seen_acc
            A_D_unseen[tno] = D_unseen_acc
            A_D_H[tno] = D_harmonic_mean
        
        for tno in range(1,task_no+1):
            if tno==1:
                sn_acc = 0
                unsn_acc = 0
                H_acc = 0
            sn_acc += A_D_seen[int(tno)]
            unsn_acc += A_D_unseen[int(tno)]
            H_acc += A_D_H[int(tno)]  
        if H_acc > best_H:
            best_H = H_acc
            best_seen = sn_acc
            best_unseen = unsn_acc
        print('Best overall accuracy at task {:d}: unseen : {:.4f}, seen : {:.4f}, H : {:.4f}'.format(task_no,best_unseen/task_no,best_seen/task_no,best_H/task_no))        
        replay_feat, replay_lab, replay_attr = replay_data(generator,discriminator,task_no,avg_feature,attr_seen_exc,seen_classes,novel_classes,no_of_replay)
    
    return replay_feat,replay_lab, replay_attr, avg_feature

def main(opt):
    data = dataloader(opt)
    discriminator = Discriminator(data.feature_size,data.att_size).cuda()
    generator= Generator(data.feature_size, data.att_size).cuda()
 
    replay_feat = None
    replay_lab = None
    replay_attr = None
    avg_feature = None

    for task_no in range(1,opt.num_tasks+1):
        replay_feat,replay_lab, replay_attr, avg_feature = train(task_no,opt.Neighbors,discriminator,generator,data,opt.seen_classes,opt.novel_classes,replay_feat,replay_lab,replay_attr,feature_size=opt.feature_size,attribute_size=opt.attribute_size,no_of_replay=opt.no_of_replay,dlr=opt.d_lr,glr=opt.g_lr,batch_size=opt.batch_size,unsn_batch_size=opt.unsn_batch_size,epochs=opt.epochs,lambda1=opt.t,alpha=opt.alpha,beta=opt.beta,avg_feature = avg_feature,all_classes=opt.all_classes,num_tasks=opt.num_tasks) 

def replay_data(generator,discriminator,task_no,avg_feature,all_attributes,seen_classes,novel_classes,no_of_replay):
        with torch.no_grad():
            lab_list = np.arange((seen_classes)*task_no)
            search_space = (seen_classes)*task_no
            _att_sn = all_attributes.unsqueeze(0).repeat([no_of_replay,1,1])
            features = discriminator(_att_sn)
            features_norm = F.normalize(features, p=2, dim=-1, eps=1e-12)
            avg_feature_1 = avg_feature.repeat(no_of_replay,1,1)
            average_features_norm = F.normalize(avg_feature_1, p=2, dim=-1, eps=1e-12)
            for i in range(len(lab_list)):
                input_att = all_attributes[lab_list[i]].repeat(no_of_replay,1)
                correct_lab = lab_list[i].repeat(no_of_replay,0)
                noise = torch.FloatTensor(no_of_replay, input_att.shape[1]).cuda()
                noise.normal_(0, 1)
                gen_fea = generator(noise,input_att) 
                gen_fea_rep = gen_fea.unsqueeze(1).repeat(1,search_space, 1)
                gen_fea_norm = F.normalize(gen_fea_rep, p=2, dim=-1, eps=1e-12)
                mean_cosine_sim = F.cosine_similarity(average_features_norm, gen_fea_norm, dim=-1)
                semantic_cosine_sim = F.cosine_similarity(features_norm, gen_fea_norm, dim=-1)
                pred1 = torch.argmax(mean_cosine_sim.squeeze(), 1) 
                pred2 = torch.argmax(semantic_cosine_sim.squeeze(), 1) 
                if i == 0:
                    loc1 = (lab_list[i]==pred1.cpu())
                    loc2 = (lab_list[i]==pred2.cpu())
                    fair_features = gen_fea[loc1==loc2]
                    fair_labels = pred1[loc1==loc2]   
                    fair_attributes = input_att[loc1==loc2]  
                else:
                    fair_features = torch.cat((fair_features,gen_fea[loc1==loc2]),0)
                    fair_labels = torch.cat((fair_labels,pred1[loc1==loc2]),0)
                    fair_attributes = torch.cat((fair_attributes,input_att[loc1==loc2]),0)
        fair_labels_1 = torch.reshape(fair_labels.clone().detach(),(fair_labels.shape[0],1)) 
        return fair_features.cpu(),fair_labels_1.cpu(), fair_attributes.cpu()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('please use GPU!')
        exit()
    cwd=os.path.dirname(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AWA1')
    parser.add_argument('--seen_classes', default=8)
    parser.add_argument('--novel_classes', default=2)
    parser.add_argument('--num_tasks', default=5)
    parser.add_argument('--all_classes', default=50)  
    parser.add_argument('--feature_size', default=2048)
    parser.add_argument('--attribute_size', default=85)
    parser.add_argument('--no_of_replay', default=300)
    parser.add_argument('--data_dir', default='../data')
    parser.add_argument('--d_lr', type=float, default=0.005)
    parser.add_argument('--g_lr', type=float, default=0.005)
    parser.add_argument('--t', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--Neighbors', type=int, default=3)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--unsn_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--dataroot', default=cwd+'/data', help='path to dataset')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')

    opt = parser.parse_args()
    main(opt)