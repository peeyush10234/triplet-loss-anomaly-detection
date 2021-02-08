import data_loader
import numpy as np
from torch import nn 
import torch 

class eval_accuracy():
    def __init__(self, df_train, df_val, model, transform, device):
        self.df_train = df_train 
        self.df_val = df_val 
        self.model = model
        self.transform = transform
        print(self.transform)
        self.device = device
    
    def get_average_embd(self):
        normal_df = self.df_train.loc[self.df_train['label']=='OK']
        anomaly_df = self.df_train.loc[self.df_train['label']=='NG']
        
        normal_dataset = data_loader.data_loader(normal_df).generate_data_loader(normal_df, None, False, 1, False, self.transform, False)
        anomaly_dataset = data_loader.data_loader(anomaly_df).generate_data_loader(anomaly_df, None, False, 1, False, self.transform, False)

        # normal_dataset = torch.utils.data.DataLoader(OnoDataset(normal_df, transform), batch_size=64)
        # anomaly_dataset = torch.utils.data.DataLoader(OnoDataset(anomaly_df, transform), batch_size=64)

        flag= True
        normal_embd = None
        anomaly_embd = None
        for data in normal_dataset:
            data = data.to(self.device)
            embd, output = self.model(data)
            embd = embd.cpu().detach().numpy()
            if flag:
                normal_embd = embd
                flag = False
            else:
                normal_embd = np.concatenate((normal_embd, embd))
        flag=True
        self.normal_embd = np.mean(normal_embd,axis=0)
        for data in anomaly_dataset:
            data = data.to(self.device)
            embd, output = self.model(data)
            embd = embd.cpu().detach().numpy()
            if flag:
                anomaly_embd = embd
                flag = False
            else:
                anomaly_embd = np.concatenate((anomaly_embd, embd))
        self.anomaly_embd = np.mean(anomaly_embd,axis=0)
    
    def calc_acc(self):
        normal_avg_embd = torch.tensor(self.normal_embd).to(self.device)
        anomaly_avg_embd = torch.tensor(self.anomaly_embd).to(self.device)
        normal_df = self.df_val.loc[self.df_val['label']=='OK']
        anomaly_df = self.df_val.loc[self.df_val['label']=='NG']
        test_loader_normal = data_loader.data_loader(normal_df).generate_data_loader(normal_df, None, False, 1, False, self.transform, False)
        test_loader_anomaly = data_loader.data_loader(anomaly_df).generate_data_loader(anomaly_df, None, False, 1, False, self.transform, False)
        d = nn.PairwiseDistance(p=2)

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for idx in test_loader_anomaly:
            idx = idx.to(self.device)
            embd, out = self.model(idx)
            loss_normal = d(embd, normal_avg_embd)
            loss_anomaly = d(embd, anomaly_avg_embd)
            if loss_anomaly<loss_normal:
                true_pos+=1
            else:
                false_neg+=1
        
        for idx in test_loader_normal:
            idx = idx.to(self.device)
            embd, out = self.model(idx)
            loss_normal = d(embd, normal_avg_embd)
            loss_anomaly = d(embd, anomaly_avg_embd)
            if loss_anomaly>loss_normal:
                true_neg+=1
            else:
                false_pos+=1

        return true_pos, true_neg, false_pos, false_neg
