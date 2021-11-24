import numpy as np
import scipy.io as sio
from scipy.stats.stats import scoreatpercentile
import torch
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import sys


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
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1      
        train_loc = matcontent['train_loc'].squeeze() - 1               
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1            
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1        
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1      
        self.attribute = torch.from_numpy(matcontent['att'].T).float() 
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()            
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.unsn_data_for_plot_1 = self.test_unseen_feature[self.test_unseen_label==6]
        self.unsn_data_for_plot_2 = self.test_unseen_feature[self.test_unseen_label==8]
    
    def semantic_similarity_check(self, Neighbors,num_tasks, train_text_feature, test_text_feature, train_label_seen, seen_classes,novel_classes,task_no):
        seen_similarity_matric = cosine_similarity(train_text_feature, train_text_feature)
        self.idx_mat = np.argsort(-1 * seen_similarity_matric, axis=1)
        self.idx_mat = self.idx_mat[:, 0:Neighbors] 
        self.semantic_similarity_seen = np.zeros((seen_similarity_matric.shape[0], Neighbors))
        for i in range(seen_similarity_matric.shape[0]):
                for j in range(Neighbors):
                    self.semantic_similarity_seen [i,j] = seen_similarity_matric[i, self.idx_mat[i,j]] 
        unseen_similarity_matric = cosine_similarity(test_text_feature, train_text_feature)
        self.unseen_idx_mat = np.argsort(-1 * unseen_similarity_matric, axis=1)
        self.unseen_idx_mat = self.unseen_idx_mat[:, 0:Neighbors]
        self.semantic_similarity_unseen = np.zeros(( unseen_similarity_matric.shape[0], Neighbors))
        for i in range(unseen_similarity_matric.shape[0]):
            for j in range(Neighbors):
                self.semantic_similarity_unseen [i,j] = unseen_similarity_matric[i, self.unseen_idx_mat[i,j]] 

    def task_train_data(self,task_no,seen_classes,all_classes,novel_classes,num_tasks,Neighbors):
        if task_no==1:
            self.lab_list = self.seenclasses
            self.seencount = 0
        else:
            self.lab_list = self.lab_list[seen_classes:]
            
        lab_list = self.lab_list[0:seen_classes]
        for i in range(len(lab_list)-1):
            idx = np.where(self.train_label == lab_list[i])
            train_feat_seen_1 = self.train_feature[idx]
            train_label_seen_1 = np.array(self.seencount).repeat(self.train_label[idx].shape[0])
            train_att_seen_1 = self.attribute[self.train_label[idx]]
            self.seencount += 1
            if i==0:
                if (task_no==1):
                    train_feat_seen = train_feat_seen_1
                    train_label_seen = train_label_seen_1
                    train_att_seen = train_att_seen_1
                else:
                    train_feat_seen = np.concatenate((train_feat_seen_1,self.extra_feat))
                    train_label_seen = np.concatenate((train_label_seen_1,self.extra_label))
                    train_att_seen = np.concatenate((train_att_seen_1,self.extra_att))
            else:     
                train_feat_seen = np.concatenate((train_feat_seen,train_feat_seen_1))
                train_label_seen = np.concatenate((train_label_seen,train_label_seen_1))
                train_att_seen = np.concatenate((train_att_seen,train_att_seen_1))
        
        idx = np.where(self.train_label == lab_list[-1])
        self.extra_feat = self.train_feature[idx]
        self.extra_label = np.array(self.seencount).repeat(self.train_label[idx].shape[0])
        self.extra_att = self.attribute[self.train_label[idx]]
        self.seencount += 1
        if(task_no==num_tasks):
            train_feat_seen = np.concatenate((train_feat_seen,self.extra_feat))
            train_label_seen = np.concatenate((train_label_seen,self.extra_label))
            train_att_seen = np.concatenate((train_att_seen,self.extra_att))
            train_text_feat = self.attribute[self.seenclasses[0:seen_classes*task_no]] 
            test_text_feat = self.attribute[self.unseenclasses[0:novel_classes*task_no]]
            self.semantic_similarity_check(Neighbors,num_tasks, train_text_feat, test_text_feat,train_label_seen, seen_classes,novel_classes,task_no)
        else:
            train_text_feat = self.attribute[self.seenclasses[0:(seen_classes*task_no)-1]] 
            test_text_feat = np.concatenate((self.attribute[self.seenclasses[(seen_classes*task_no)-1:(seen_classes*task_no)]],self.attribute[self.unseenclasses[0:novel_classes*task_no]]))
            self.semantic_similarity_check(Neighbors,num_tasks, train_text_feat, test_text_feat,train_label_seen, seen_classes,novel_classes,task_no)
            
        train_label_seen = torch.reshape(torch.tensor(train_label_seen),(train_label_seen.shape[0],1))
        return torch.tensor(train_feat_seen), train_label_seen,torch.tensor(train_att_seen)


    def attribute_mapping(self,seen_classes,novel_classes,task_no):
        for i in range(seen_classes*task_no):
            if i == 0:
                unique_attribute = self.attribute[self.seenclasses[i]]
            else:
                unique_attribute = torch.cat((unique_attribute,self.attribute[self.seenclasses[i]]))
        for j in range(novel_classes*task_no):
            unique_attribute = torch.cat((unique_attribute,self.attribute[self.unseenclasses[j]]))
        unique_attributes = torch.reshape(unique_attribute,[(seen_classes+novel_classes)*task_no,-1])
        return unique_attributes

    def train_attribute_seen_exclusive(self,seen_classes,task_no):
        unique_attributes = self.attribute[self.seenclasses[0:seen_classes*task_no]] 
        unique_attributes = unique_attributes.cuda()
        return unique_attributes

    def train_attribute_unseen_exclusive(self,novel_classes,task_no):
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
        test_seen_l = {}
        test_seen_a = {}
        test_unseen_l = {}
        test_unseen_a = {}   
        for tno in range(1,task_no+1):
            dup_seenclasses = self.seenclasses
            dup_unseenclasses = self.unseenclasses           
            self.testseen = 0
            self.testunseen = seen_classes*tno
            lab_list = dup_seenclasses[:seen_classes*tno]      
            for i in range(len(lab_list)-1):
                idx = np.where(self.test_seen_label == lab_list[i])
                test_label_seen_1 = np.array(self.testseen).repeat(self.test_seen_label[idx].shape[0])
                test_att_seen_1 = self.attribute[self.test_seen_label[idx]]
                self.testseen += 1
                if  i==0:
                    test_label_seen = test_label_seen_1
                    test_att_seen = test_att_seen_1
                else:
                    test_label_seen = np.concatenate((test_label_seen,test_label_seen_1))
                    test_att_seen = np.concatenate((test_att_seen,test_att_seen_1))
            idx = np.where(self.test_seen_label == lab_list[-1])
            self.extra_label_test = np.array(self.testseen).repeat(self.test_seen_label[idx].shape[0])
            self.extra_att_test = self.attribute[self.test_seen_label[idx]]
            if(tno==num_tasks):
                test_label_seen = np.concatenate((test_label_seen,self.extra_label_test))
                test_att_seen = np.concatenate((test_att_seen,self.extra_att_test))
            lab_list_un = dup_unseenclasses[:novel_classes*tno]
            for i in range(len(lab_list_un)):
                idx = np.where(self.test_unseen_label == lab_list_un[i])
                test_label_unseen_1 = np.array(self.testunseen).repeat(self.test_unseen_label[idx].shape[0])
                test_att_unseen_1 = (self.attribute[self.test_unseen_label[idx]])
                self.testunseen += 1
                if i==0:
                    if (tno<num_tasks):
                        test_label_unseen = np.concatenate((self.extra_label_test,test_label_unseen_1))
                        test_att_unseen = np.concatenate((self.extra_att_test,test_att_unseen_1))
                    else:
                        test_label_unseen = test_label_unseen_1
                        test_att_unseen =  test_att_unseen_1
                else:
                    test_label_unseen = np.concatenate((test_label_unseen,test_label_unseen_1))
                    test_att_unseen = np.concatenate((test_att_unseen,test_att_unseen_1))
            test_seen_l[tno] = torch.tensor(test_label_seen)
            test_seen_a[tno] = torch.tensor(test_att_seen)
            test_label_seen = None
            test_att_seen = None
            test_unseen_l[tno] = torch.tensor(test_label_unseen)
            test_unseen_a[tno] = torch.tensor(test_att_unseen)
            test_label_unseen = None
            test_att_unseen = None
        return test_unseen_l,test_unseen_a  

    def task_test_data_(self,task_no,seen_classes,all_classes,novel_classes,num_tasks): 
        dup_seenclasses = self.seenclasses
        dup_unseenclasses = self.unseenclasses
        self.testseen = 0
        self.testunseen = seen_classes*task_no
        lab_list = dup_seenclasses[:seen_classes*task_no]   
        for i in range(len(lab_list)-1):
            idx = np.where(self.test_seen_label == lab_list[i])
            test_feat_seen_1 = self.test_seen_feature[idx]
            test_label_seen_1 = np.array(self.testseen).repeat(self.test_seen_label[idx].shape[0])
            test_att_seen_1 = self.attribute[self.test_seen_label[idx]]
            self.testseen += 1
            if  i==0:
                test_feat_seen = test_feat_seen_1
                test_label_seen = test_label_seen_1
                test_att_seen = test_att_seen_1
            else:
                test_feat_seen = np.concatenate((test_feat_seen,test_feat_seen_1))
                test_label_seen = np.concatenate((test_label_seen,test_label_seen_1))
                test_att_seen = np.concatenate((test_att_seen,test_att_seen_1))
        idx = np.where(self.test_seen_label == lab_list[-1])
        self.extra_feat_test = self.test_seen_feature[idx]
        self.extra_label_test = np.array(self.testseen).repeat(self.test_seen_label[idx].shape[0])
        self.extra_att_test = self.attribute[self.test_seen_label[idx]]
        if(task_no==num_tasks):
            test_feat_seen = np.concatenate((test_feat_seen,self.extra_feat_test))
            test_label_seen = np.concatenate((test_label_seen,self.extra_label_test))
            test_att_seen = np.concatenate((test_att_seen,self.extra_att_test))
        lab_list_un = dup_unseenclasses[:novel_classes*task_no]
        for i in range(len(lab_list_un)):
            idx = np.where(self.test_unseen_label == lab_list_un[i])
            test_feat_unseen_1 = self.test_unseen_feature[idx]
            test_label_unseen_1 = np.array(self.testunseen).repeat(self.test_unseen_label[idx].shape[0])
            test_att_unseen_1 = (self.attribute[self.test_unseen_label[idx]])
            self.testunseen += 1
            if i==0:
                if (task_no<num_tasks):
                    test_feat_unseen = np.concatenate((self.extra_feat_test,test_feat_unseen_1))
                    test_label_unseen = np.concatenate((self.extra_label_test,test_label_unseen_1))
                    test_att_unseen = np.concatenate((self.extra_att_test,test_att_unseen_1))
                else:
                    test_feat_unseen = test_feat_unseen_1
                    test_label_unseen = test_label_unseen_1
                    test_att_unseen =  test_att_unseen_1
            else:
                test_feat_unseen = np.concatenate((test_feat_unseen,test_feat_unseen_1))
                test_label_unseen = np.concatenate((test_label_unseen,test_label_unseen_1))
                test_att_unseen = np.concatenate((test_att_unseen,test_att_unseen_1))
        return torch.tensor(test_feat_seen),torch.tensor(test_label_seen),torch.tensor(test_att_seen),torch.tensor(test_feat_unseen),torch.tensor(test_label_unseen),torch.tensor(test_att_unseen)
        
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