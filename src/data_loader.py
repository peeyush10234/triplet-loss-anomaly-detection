import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader,Dataset

class data_loader():
    def __init__(self, image_df):
        self.image_df = image_df

    def create_train_val_split(self, val_split = 0.2):
        self.image_df['img_name'] = self.image_df['image_path'].map(lambda x: ('-'.join(x.split('/')[-1].split('-')[:-1])))
        print(self.image_df['img_name'])
        label_gr = self.image_df.groupby(['label'])
        ng_df = label_gr.get_group('NG')
        ok_df = label_gr.get_group('OK')
        ng_img_names = list(set(ng_df['img_name'].values))
        ok_img_names = list(set(ok_df['img_name'].values))
        r_s = random.sample(range(len(ng_img_names)), int(len(ng_img_names)*0.2))
        val_img_names_ng = [ng_img_names[x] for x in range(len(ng_img_names)) if x in r_s]
        val_ng_df = ng_df.loc[ng_df['img_name'].map(lambda x : x in val_img_names_ng)]
        train_ng_df = ng_df.loc[ng_df['img_name'].map(lambda x : x not in val_img_names_ng)]
        r_s = random.sample(range(len(ok_img_names)), int(len(ok_img_names)*0.2))
        val_img_names_ok = [ok_img_names[x] for x in range(len(ok_img_names)) if x in r_s]
        val_ok_df = ok_df.loc[ok_df['img_name'].map(lambda x : x in val_img_names_ok)]
        train_ok_df = ok_df.loc[ok_df['img_name'].map(lambda x : x not in val_img_names_ok)]
        df_val = pd.concat([val_ok_df, val_ng_df])
        df_train = pd.concat([train_ok_df, train_ng_df])
        return df_train, df_val
    
    def generate_data_loader(self, df_train, df_val=None, create_triplets=True, batch_size=64, shuffle=True, 
                                transform=None, training=True):
        self.df_train = df_train
        self.df_val = df_val
        self.transform = transform
        self.create_triplets = create_triplets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training

        if self.create_triplets:
            df_dataset = onoDatasetTriplets(self.df_train, self.df_val, self.transform, self.training)
        else:
            df_dataset = onoDataset(self.df_train, self.transform)
        
        data_loader = torch.utils.data.DataLoader(df_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return data_loader

class onoDataset(Dataset):
  def __init__(self, df, transform=None):
    self.df = df
    self.transform = transform

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    image = Image.open(self.df.iloc[idx]['image_path'])

    if self.transform!=None:
      image = self.transform(image)
    
    return image


class onoDatasetTriplets(Dataset):
  def __init__(self, df_train, df_val, transform=None, training=True):
    self.df_train = df_train
    self.df_val = df_val
    self.df_normal = self.df_train.loc[self.df_train['label'] == 'OK']
    self.df_anomaly = self.df_train.loc[self.df_train['label'] == 'NG']
    self.transform = transform
    if training == False:
      self.df_normal = self.df_val.loc[self.df_val['label'] == 'OK']
      self.df_anomaly = self.df_val.loc[self.df_val['label'] == 'NG']

  def __len__(self):
    return self.df_normal.shape[0]

  def __getitem__(self, idx):
    pos_idx = random.randint(0, self.df_normal.shape[0]-1)
    neg_idx = random.randint(0, self.df_anomaly.shape[0]-1)
    anchor = Image.open(self.df_normal.iloc[idx]['image_path'])
    positive = Image.open(self.df_normal.iloc[pos_idx]['image_path'])
    negative = Image.open(self.df_anomaly.iloc[neg_idx]['image_path'])

    if self.transform!=None:
      anchor  = self.transform(anchor)
      positive = self.transform(positive)
      negative = self.transform(negative)
    
    return anchor, positive, negative
