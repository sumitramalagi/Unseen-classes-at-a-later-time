import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import argparse
import os
#### Specify the appropriate dataloader here ### 
from Dataloader_dynamic import DATA_LOADER as dataloader


cwd=os.path.dirname(os.getcwd())

torch.manual_seed(55)
torch.cuda.manual_seed(55)
      
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


def test(task_no,discriminator,data,seen_classes,novel_classes,batch_size=64,all_classes=None,num_tasks=None):    

    # whole_attr_seen = whole_attr_seen.cuda()  

    best_seen = 0
    best_unseen = 0
    best_H = 0
    A_D_seen = {}
    A_D_unseen = {}
    A_D_H = {}

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

def main(opt):
    data = dataloader(opt)
    discriminator = Discriminator(data.feature_size,data.att_size)
    checkptname='checkpoint'+'_'+str(opt.num_tasks)+'.pt'
    checkpnt=torch.load(checkptname, map_location=lambda storage, loc: storage)
    discriminator_state = checkpnt['discriminator']
    discriminator.load_state_dict(discriminator_state)
    discriminator=discriminator.cuda()
    test(opt.num_tasks,discriminator,data,opt.seen_classes,opt.novel_classes,batch_size=opt.batch_size,all_classes=opt.all_classes,num_tasks=opt.num_tasks) 



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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--unsn_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--dataroot', default=cwd+'/data', help='path to dataset')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')

    opt = parser.parse_args()
    main(opt)