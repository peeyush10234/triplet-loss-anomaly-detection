import sys
sys.path.append('..')
import model
import loss_func
import data_loader
import pickle
import numpy as np
from torch import nn
import torch
import pandas as pd
from config import configs
import torchvision.transforms as transforms


class train_val():
    def __init__(self, train_loader, val_loader, model, optimizer ,num_epoch):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.distance = nn.MSELoss()
        self.triplet_loss = loss_func.LosslessTripletLoss()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def valid_fn(self):
        self.model.eval()
        total_rc_loss=0
        total_triplet_loss=0
        ct = 0
        for anchor, pos, neg in self.val_loader:
            anchor = anchor.to(self.device)
            pos = pos.to(self.device)
            neg = pos.to(self.device)
            anchor_e, output = self.model(anchor)
            pos_e, output1 = self.model(pos)
            neg_e, output2 = self.model(neg)
            rc_loss = self.distance(anchor, output)
            t_loss = self.triplet_loss(anchor_e, pos_e, neg_e)
            total_rc_loss+=rc_loss.item()
            total_triplet_loss+=t_loss.item()
            ct+=1
        print(ct)
        return total_rc_loss/ct, total_triplet_loss/ct
    
    def save_model_loss(self, loss_dict, epoch):
        model_save_path = configs.saved_model_dir / f'ae_triplet_model_{epoch}.pth'
        model_save_path_latest = configs.saved_model_dir / f'ar_triplet_model_latest.pth'
        loss_dict_save_path = configs.saved_model_dir / f'loss_dict_{epoch}.pkl'
        loss_dict_save_path_latest = configs.saved_model_dir / f'loss_dict_latest.pkl'
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_save_path)
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_save_path_latest)
        
        with open(loss_dict_save_path, 'w') as f_w:
            pickle.dump(loss_dict, f_w)
        with open(loss_dict_save_path_latest, 'w') as f_w:
            pickle.dump(loss_dict, f_w)
        
    def train_func(self):
        loss_list_g = []
        epoch_loss_total = []
        epoch_loss_lrc = []
        epoch_loss_lt = []
        epoch_val_loss_lrc = []
        epoch_val_loss_lt = []
        step_loss_total = []
        step_loss_lrc = []
        step_loss_lt = []
        epoch_dn_loss = []
        epoch_dp_loss = []
        step_dn_loss = []
        step_dp_loss = []
        number_of_zeros_per_batch = []
        total_steps = 0
        loss_dict = {}

        for epoch in range(self.num_epoch):
            total_rc_loss = 0.0
            total_triplet_loss = 0.0
            loss_list = []
            total_dp = 0.0
            total_dn = 0.0
            ct = 0
            loss_t = 0.0
            for  anchor, pos, neg in self.train_loader:
                anchor = anchor.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)
                # print(anchor.requires_grad, pos.requires_grad, neg.requires_grad)
                anchor_e, output = self.model(anchor)
                pos_e, output1 = self.model(pos)
                neg_e, output2 = self.model(neg)
                # if total_steps in [0, 200, 9000]:
                #   save_csv(anchor_e, pos_e, neg_e, total_steps)
                d = nn.PairwiseDistance(p=2)
                dist_p = d(anchor_e,pos_e) 
                dist_n = d(anchor_e,neg_e)
                distance_l = dist_p - dist_n
                batch_wise_loss = torch.max(distance_l, torch.zeros_like(distance_l))
                batch_np = batch_wise_loss.cpu().detach().numpy()
                number_of_zeros_per_batch.append(np.where( batch_np == 0.0)[0].size)     
                loss_t+=torch.mean(batch_wise_loss).item()
                # print(loss)
                # print(torch.mean(dist_p), )
                total_dp+=torch.mean(dist_p)
                total_dn+=torch.mean(dist_n)
                rc_loss = self.distance(anchor, output)
                t_loss = self.triplet_loss(anchor_e, pos_e, neg_e)
                loss = rc_loss + t_loss
                loss.backward()
                self.optimizer.step()
                total_rc_loss+=rc_loss.item()
                total_triplet_loss+=t_loss.item()
                step_loss_lrc.append(rc_loss.item())
                step_loss_lt.append(t_loss.item())
                step_loss_total.append(rc_loss.item() + t_loss.item())
                step_dp_loss.append(torch.mean(dist_p))
                step_dn_loss.append(torch.mean(dist_n))
                self.optimizer.zero_grad()
                ct+=1
                total_steps+=1
                loss_list.append(rc_loss.item())
            loss_list_g = loss_list
            epoch_loss_total.append((total_rc_loss+total_triplet_loss)/ct)
            epoch_loss_lrc.append(total_rc_loss/ct)
            epoch_loss_lt.append(total_triplet_loss/ct)
            epoch_dn_loss.append(total_dn/ct)
            epoch_dp_loss.append(total_dp/ct)
            print(loss_t/ct)
            print('epoch [{}/{}], triplet_loss:{:.4f}, rc_loss:{:.4f}'.format(epoch+1, self.num_epoch, total_triplet_loss/ct, total_rc_loss/ct))
            val_lrc, val_lt = self.valid_fn()
            print(val_lrc, val_lt)
            epoch_val_loss_lrc.append(val_lrc)
            epoch_val_loss_lt.append(val_lt)
            
            loss_dict['epoch_loss_total'] = epoch_loss_total
            loss_dict['epoch_loss_lrc'] = epoch_loss_lrc
            loss_dict['epoch_loss_lt'] = epoch_loss_lt
            loss_dict['epoch_val_loss_lrc'] = epoch_val_loss_lrc
            loss_dict['epoch_val_loss_lt'] = epoch_val_loss_lt
            loss_dict['step_loss_total'] = step_loss_total
            loss_dict['step_loss_lrc'] = step_loss_lrc
            loss_dict['step_loss_lt'] = step_loss_lt
            loss_dict['epoch_dn_loss'] = epoch_dn_loss
            loss_dict['epoch_dp_loss'] = epoch_dp_loss
            loss_dict['step_dn_loss'] = step_dn_loss
            loss_dict['step_dp_loss'] = step_dp_loss

            if True:
                self.save_model_loss(loss_dict, epoch)

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    num_epochs = 100

    image_df = pd.read_csv('../crop_image_paths.csv')
    data_loader_obj = data_loader.data_loader(image_df)
    df_train, df_val = data_loader_obj.create_train_val_split(val_split=0.2)
    train_loader = data_loader_obj.generate_data_loader(df_train, df_val, True, 64, transform, True)
    val_loader = data_loader_obj.generate_data_loader(df_train, df_val, True, 64, transform, False)
    
    model = model.CnnAutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    train_val_obj = train_val(train_loader, val_loader, model, optimizer, num_epochs)
    train_val_obj.train_func()