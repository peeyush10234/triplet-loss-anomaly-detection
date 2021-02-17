import sys 
sys.path.append('..')
import model 
import loss_func
import data_loader
from config import configs
import train
import argparse
from sklearn.model_selection import StratifiedKFold 
import pandas as pd
import train
import torchvision.transforms as transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', help = 'Number of folds', type=int, default=5)
    args = parser.parse_args()
    augmented_image_df = pd.read_csv('../augmented_images.csv')
    augment_image_gb = augmented_image_df.groupby('img_name')
    img_name = list(set(augmented_image_df['img_name'].values))

    label_list = []
    for img in img_name:
        label = augment_image_gb.get_group(img).iloc[0]['label']
        label_list.append([img,label])
    
    img_name_df = pd.DataFrame(label_list, columns=['img_name', 'temp_labels'])
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    img_name_df['kfold'] = 0

    for f, (t_idx, v_idx) in enumerate(skf.split(X=img_name_df['img_name'], y=img_name_df['temp_labels'])):
        img_name_df.loc[v_idx, 'kfold'] = int(f)
    
    merge_df = pd.merge(augmented_image_df, img_name_df, on = 'img_name')
    print(merge_df.shape)

    transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    # model = model.CnnAutoEncoder()

    for fold in range(args.folds):
        val_fold = fold

        df_val = merge_df[merge_df['kfold'] == val_fold]
        df_train = merge_df[merge_df['kfold'] != val_fold]

        print(df_train.shape, df_val.shape)
        cnn_model = model.CnnAutoEncoder()
        val_dict = train.train_init(num_epochs=5, learning_rate=0.0001, batch_size=64,
                    save_interval=5, transform=transform, model=cnn_model, image_df = augmented_image_df,
                    df_train=df_train, df_val=df_val)
        
        print(val_dict)