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
import eval_accuracy
import argparse
import report_generation

class train_val():
    def __init__(self, df_train, df_val, train_loader, val_loader, model, optimizer ,num_epoch, transform, save_interval, learning_rate, batch_size):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.df_train = df_train
        self.df_val = df_val
        self.model = model
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.distance = nn.MSELoss()
        self.triplet_loss = loss_func.LosslessTripletLoss()
        self.transform = transform
        self.save_interval = save_interval
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
        model_save_path_latest = configs.saved_model_dir / f'ae_triplet_model_latest.pth'
        loss_dict_save_path = configs.saved_model_dir / f'loss_dict_{epoch}.pkl'
        loss_dict_save_path_latest = configs.saved_model_dir / f'loss_dict_latest.pkl'
        model_metrics_path = configs.saved_model_dir / f'ae_triplet_metrics_{epoch}.pkl'
        model_metrics_path_latest = configs.saved_model_dir / f'ae_triplet_metrics_latest.pkl'

        true_pos, true_neg, false_pos, false_neg  = self.calc_results()
        precision = true_pos/(true_pos + false_pos)
        recall = true_pos/(true_pos + false_neg)
        print(true_pos, true_neg, false_pos, false_neg, precision, recall)
        val_dict = {
            'true_pos' : true_pos, 'true_neg' : true_neg, 'false_pos' : false_pos, 'false_neg' : false_neg,
            'precision' : precision, 'recall' : recall
        }

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
        
        with open(loss_dict_save_path, 'wb') as f_w:
            pickle.dump(loss_dict, f_w)
        with open(loss_dict_save_path_latest, 'wb') as f_w:
            pickle.dump(loss_dict, f_w)
        with open(model_metrics_path, 'wb') as f_w:
            pickle.dump(val_dict, f_w)
        with open(model_metrics_path_latest, 'wb') as f_w:
            pickle.dump(val_dict, f_w)
        
        # model_param = {
        # 'epochs' :  num_epochs,
        # 'learning_rate' : learning_rate,
        # 'batch_size' : batch_size,
        # 'optimizer' : 'Adam',
        # 'Triplet Loss Margin' : 512
        # }
        model_param = [
            [self.num_epoch, self.learning_rate, self.batch_size, 'Adam', 512]
        ]

        model_param_df = pd.DataFrame(model_param, columns=['Epochs', 'Learning Rate', 'Batch Size', 'Optimizer', 'Triplet Loss Margin'])

        rg_obj = report_generation.report_generation(self.df_train, self.df_val, loss_dict, val_dict, model_param_df)
        rg_obj.create_report_df()
        rg_obj.generate_report()

    def calc_results(self):
        eval_accuracy_obj = eval_accuracy.eval_accuracy(self.df_train, self.df_val, self.model, self.transform, self.device)
        eval_accuracy_obj.get_average_embd()
        print(eval_accuracy_obj.anomaly_embd.shape, eval_accuracy_obj.normal_embd.shape)
        return eval_accuracy_obj.calc_acc()
        
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

            if (epoch+1)%self.save_interval == 0:
                self.save_model_loss(loss_dict, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', help = 'Number of epochs for which the model will run', type=int, default=100)
    parser.add_argument('--learning_rate', help = 'Set learning rate for the model', type=float, default=1e-4)
    parser.add_argument('--image_df_path', 
        help = 'Describe the image df path which store image patha and corresponding label',
        type=str, default='../crop_image_paths.csv')
    parser.add_argument('--batch_size', help = "Set the batch size", type=int, default=64)
    parser.add_argument('--save_interval', help = 'Number of epcohs interval after which we save result and model',
        type=int, default=100)
    args = parser.parse_args()

    transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_interval = args.save_interval
    image_df = pd.read_csv(args.image_df_path)
    data_loader_obj = data_loader.data_loader(image_df)
    df_train, df_val = data_loader_obj.create_train_val_split(val_split=0.2)
    print(df_train.shape, df_val.shape)
    train_loader = data_loader_obj.generate_data_loader(df_train, df_val, True, batch_size, True, transform, True)
    val_loader = data_loader_obj.generate_data_loader(df_train, df_val, True, batch_size, True, transform, False)
    
    model = model.CnnAutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train_val_obj = train_val(df_train, df_val, train_loader, val_loader, model, optimizer, num_epochs, transform, save_interval, learning_rate, batch_size)
    train_val_obj.train_func()
    