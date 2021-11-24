import numpy as np
import scipy.io as sio
from scipy.stats.stats import scoreatpercentile
import torch
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import sys
import pdb


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0  
        self.feature_size = self.train_feature.shape[1]
        self.att_size = self.attribute.shape[1]
  
   

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        self.attribute = torch.from_numpy(matcontent['att'].T).float() 
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()      
        self.feat_train, self.feat_test, self.label_train, self.label_test = train_test_split(self.train_feature, self.train_label, test_size=0.2, random_state=42)
        
    def semantic_similarity_check(self, Neighbors, train_text_feature, test_text_feature, train_label_seen, seen_classes,all_classes,task_no,num_tasks):
        seen_similarity_matric = cosine_similarity(train_text_feature, train_text_feature)
        self.idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        self.idx_mat = self.idx_mat[:, 0:Neighbors] 
        self.semantic_similarity_seen = np.zeros((seen_classes*task_no, Neighbors))

        for i in range(seen_classes*task_no):
                for j in range(Neighbors):
                    self.semantic_similarity_seen [i,j] = seen_similarity_matric[i, self.idx_mat[i,j]]      
        if task_no < num_tasks:
            unseen_similarity_matric = cosine_similarity(test_text_feature, train_text_feature)
            self.unseen_idx_mat = np.argsort(-1 * unseen_similarity_matric, axis=1)
            self.unseen_idx_mat = self.unseen_idx_mat[:, 0:Neighbors]
            self.semantic_similarity_unseen = np.zeros((all_classes- seen_classes*task_no, Neighbors))
     
            for i in range(all_classes-seen_classes*task_no):
                for j in range(Neighbors):
                    self.semantic_similarity_unseen [i,j] = unseen_similarity_matric[i, self.unseen_idx_mat[i,j]] 

    def task_train_data(self,task_no,seen_classes,all_classes,novel_classes,num_tasks,Neighbors):
        for i in range(seen_classes*(task_no-1),seen_classes*task_no):
            idx = np.where(self.label_train == i)
            train_feat_seen_1 = self.feat_train[idx]
            train_label_seen_1 = np.array(i).repeat(self.label_train[idx].shape[0])
            train_att_seen_1 = self.attribute[self.label_train[idx]]
     
            if i==seen_classes*(task_no-1):
                train_feat_seen = train_feat_seen_1
                train_label_seen = train_label_seen_1
                train_att_seen = train_att_seen_1
            else:     
                train_feat_seen = np.concatenate((train_feat_seen,train_feat_seen_1))
                train_label_seen = np.concatenate((train_label_seen,train_label_seen_1))
                train_att_seen = np.concatenate((train_att_seen,train_att_seen_1))    
        train_text_feat = self.attribute[0:seen_classes*task_no] 
        test_text_feat = self.attribute[seen_classes*task_no:]
        self.semantic_similarity_check(Neighbors, train_text_feat, test_text_feat,train_label_seen, seen_classes,all_classes, task_no,num_tasks)
        train_label_seen = torch.reshape(torch.tensor(train_label_seen),(train_label_seen.shape[0],1))
        return torch.tensor(train_feat_seen), train_label_seen,torch.tensor(train_att_seen)


    def attribute_mapping(self,seen_classes,novel_classes,task_no):
        return self.attribute

    def train_attribute_seen_exclusive(self,seen_classes,task_no):
        unique_attributes = self.attribute[self.seenclasses[0:seen_classes*task_no]] 
        unique_attributes = unique_attributes.cuda()
        return unique_attributes

    def train_attribute_unseen_exclusive(self,task_no):
        unique_attributes = self.attribute[self.unseenclasses[novel_classes*(task_no-1):novel_classes*task_no]] 
        unique_attributes = unique_attributes.cuda()

        return unique_attributes
    
    def test_attribute_seen_exclusive(self,seen_classes,task_no):
        if task_no==1:
            self.seenattr_list = self.seenclasses
        unique_attributes = self.attribute[self.seenattr_list[0:seen_classes]] 
        self.seenattr_list = self.seenattr_list[seen_classes:]
        unique_attributes = unique_attributes.cuda()
        return unique_attributes

    def test_attribute_unseen_exclusive(self,novel_classes,task_no):
        if task_no==1:
            self.unseenattr_list = self.unseenclasses
        unique_attributes = self.attribute[self.unseenattr_list[0:novel_classes]] 
        self.unseenattr_list = self.unseenattr_list[novel_classes:]
        unique_attributes = unique_attributes.cuda()
        return unique_attributes
  
    def task_test_data(self,task_no,seen_classes,all_classes,novel_classes,num_tasks): 
        test_seen_f = {}
        test_seen_l = {}
        test_seen_a = {}
        test_unseen_f = {}
        test_unseen_l = {}
        test_unseen_a = {}
        self.testunseen = seen_classes*task_no    
        for tno in range(1,task_no+1):
            lab_list = seen_classes*task_no
           
            for i in range(lab_list):
                idx = np.where(self.label_test == i)
                test_feat_seen_1 = self.feat_test[idx]
                test_label_seen_1 = np.array(i).repeat(self.label_test[idx].shape[0])
                test_att_seen_1 = self.attribute[self.label_test[idx]]
 
                if  i==0:
                    test_feat_seen = test_feat_seen_1
                    test_label_seen = test_label_seen_1
                    test_att_seen = test_att_seen_1
                else:
                    test_feat_seen = np.concatenate((test_feat_seen,test_feat_seen_1))
                    test_label_seen = np.concatenate((test_label_seen,test_label_seen_1))
                    test_att_seen = np.concatenate((test_att_seen,test_att_seen_1))
            test_seen_f[tno] = torch.tensor(test_feat_seen)
            test_seen_l[tno] = torch.tensor(test_label_seen)
            test_seen_a[tno] = torch.tensor(test_att_seen)
            test_feat_seen = None
            test_label_seen = None
            test_att_seen = None
            for i in range(seen_classes*task_no,all_classes):
                idx = np.where(self.label_test == i)
                test_feat_unseen_1 = self.feat_test[idx]
                test_label_unseen_1 = np.array(i).repeat(self.label_test[idx].shape[0])
                test_att_unseen_1 = (self.attribute[self.label_test[idx]])

                if i==seen_classes*task_no:
                    test_feat_unseen = test_feat_unseen_1
                    test_label_unseen = test_label_unseen_1
                    test_att_unseen =  test_att_unseen_1
                else:
                    test_feat_unseen = np.concatenate((test_feat_unseen,test_feat_unseen_1))
                    test_label_unseen = np.concatenate((test_label_unseen,test_label_unseen_1))
                    test_att_unseen = np.concatenate((test_att_unseen,test_att_unseen_1))
            test_unseen_f[tno] = torch.tensor(test_feat_unseen)
            test_unseen_l[tno] = torch.tensor(test_label_unseen)
            test_unseen_a[tno] = torch.tensor(test_att_unseen)
            test_feat_unseen = None
            test_label_unseen = None
            test_att_unseen = None  
        return test_unseen_l,test_unseen_a

    def task_test_data_(self,task_no,seen_classes,all_classes,novel_classes,num_tasks): 
        self.testseen = 0
        self.testunseen = seen_classes*task_no
        lab_list = seen_classes*task_no   
        for i in range(lab_list):
            idx = np.where(self.label_test == i)
            test_feat_seen_1 = self.feat_test[idx]
            test_label_seen_1 = np.array(self.testseen).repeat(self.label_test[idx].shape[0])
            test_att_seen_1 = self.attribute[self.label_test[idx]]
            self.testseen += 1
            if  i==0:
                test_feat_seen = test_feat_seen_1
                test_label_seen = test_label_seen_1
                test_att_seen = test_att_seen_1
            else:
                test_feat_seen = np.concatenate((test_feat_seen,test_feat_seen_1))
                test_label_seen = np.concatenate((test_label_seen,test_label_seen_1))
                test_att_seen = np.concatenate((test_att_seen,test_att_seen_1))
        for i in range(seen_classes*task_no,all_classes):
            idx = np.where(self.label_test == i)
            test_feat_unseen_1 = self.feat_test[idx]
            test_label_unseen_1 = np.array(self.testunseen).repeat(self.label_test[idx].shape[0])
            test_att_unseen_1 = (self.attribute[self.label_test[idx]])
            self.testunseen += 1
            if i==seen_classes*task_no:
                test_feat_unseen = test_feat_unseen_1
                test_label_unseen = test_label_unseen_1
                test_att_unseen =  test_att_unseen_1
            else:
                test_feat_unseen = np.concatenate((test_feat_unseen,test_feat_unseen_1))
                test_label_unseen = np.concatenate((test_label_unseen,test_label_unseen_1))
                test_att_unseen = np.concatenate((test_att_unseen,test_att_unseen_1))          
        if task_no == num_tasks:
            test_feat_unseen = None
            test_label_unseen = None
            test_att_unseen = None
        else:
            test_feat_unseen = torch.tensor(test_feat_unseen)
            test_label_unseen = torch.tensor(test_label_unseen)
            test_att_unseen = torch.tensor(test_att_unseen)
        return torch.tensor(test_feat_seen),torch.tensor(test_label_seen),torch.tensor(test_att_seen),test_feat_unseen,test_label_unseen,test_att_unseen

    def next_batch_seen(self,ninstance,pos_labels,neg_labels,seen_feat,whole_attr,whole_labels):
        for i in range(pos_labels.shape[0]):
            c1 = np.where(pos_labels[i]==whole_labels)
            np.random.shuffle(c1[0])
            if i==0:
                batch_label = whole_labels[c1[0]][0:ninstance]        
                batch_feature = seen_feat[c1[0]][0:ninstance]
                batch_attr = whole_attr[c1[0]][0:ninstance]
            else:
                batch_label = torch.cat((batch_label,whole_labels[c1[0]][0:ninstance]))
                batch_feature = torch.cat((batch_feature,seen_feat[c1[0]][0:ninstance]))
                batch_attr = torch.cat((batch_attr,whole_attr[c1[0]][0:ninstance]))
        for i in range(neg_labels.shape[0]):
            c1 = np.where(neg_labels[i]==whole_labels)
            np.random.shuffle(c1[0])
            if i==0:
                batch_label_neg = whole_labels[c1[0][0:ninstance]]
                batch_feature_neg = seen_feat[c1[0][0:ninstance]]
                batch_attr_neg = whole_attr[c1[0][0:ninstance]]
            else:
                batch_label_neg = torch.cat((batch_label_neg,whole_labels[c1[0][0:ninstance]]))
                batch_feature_neg = torch.cat((batch_feature_neg,seen_feat[c1[0][0:ninstance]]))
                batch_attr_neg = torch.cat((batch_attr_neg,whole_attr[c1[0][0:ninstance]]))   
        return  [batch_label,batch_attr, batch_feature] , [batch_label_neg, batch_attr_neg,batch_feature_neg ]
    
    
    def test_batch_unseen(self,nsamp,ninstance,attr_unseen,label_unseen):
        a1 = np.unique(label_unseen)
        np.random.shuffle(a1)
        b1 = a1[0:nsamp]       
        for i in range(nsamp):
            c1 = np.where(b1[i]==label_unseen)
            if i==0:
                batch_label = label_unseen[c1[0][0:1]]
                batch_attr = attr_unseen[c1[0][0:1]]
            else:
                batch_label = np.concatenate((batch_label,label_unseen[c1[0][0:1]]))
                batch_attr = torch.cat((batch_attr,attr_unseen[c1[0][0:1]]))
        return   [torch.tensor(batch_label), batch_attr]

    def next_batch(self,batch_size,label,feature,attributes):       
        idx = torch.randperm(feature.shape[0])[0:batch_size]
        batch_feature = feature[idx]
        batch_label = label[idx]
        batch_attr = attributes[idx]
        return batch_feature, batch_label, batch_attr