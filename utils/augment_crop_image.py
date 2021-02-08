import sys
sys.path.append('..')
from config import configs
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import os
class augment_crop_image():
    def __init__(self, image_df, augmentations):
        self.df = image_df
        self.augmentations = augmentations
    
    def augment_image(self):
        angles = self.augmentations['angles']
        flips = self.augmentations['flip']
        for index, row in self.df.iterrows():
            image_name = row['image_path'].split('/')[-1].split('.')[0]
            image_path = configs.root_dir / row['image_path']
            img = cv2.imread(str(image_path))
            ct=0
            for angle in angles:
                for flip in flips:
                    img_aug = cv2.rotate(img, angle) if angle!=None else img
                    img_aug = cv2.flip(img_aug, flip) if flip!=None else img_aug
                    aug_img_name = f'{image_name}-{row.label}-{ct}.bmp'
                    aug_image_path = configs.augmented_image_dir / aug_img_name
                    plt.imsave(aug_image_path, img_aug)
                    ct+=1
    
    def crop_save_image(self, file_path):
        img = cv2.imread(file_path)
        filename = file_path.split('/')[-1].split('.')[0]
        output = img.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT ,1,2000)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x,y,r) in circles:
                crop_img = output[y-(r+10):y+(r+10), x-(r+10):x+(r+10)]
                crop_image_path = f'{configs.cropped_image_dir}/{filename}.bmp'
                plt.imsave(crop_image_path,crop_img)
                return crop_image_path
        return None

    def crop_image_dir(self):
        augment_image_dir = configs.augmented_image_dir
        image_list = []
        # for image_path in augment_image_dir.glob('*.bmp'):
        #     image_path = str(image_path)
        #     label = image_path.split('-')[1]
        #     crop_image_path = self.crop_save_image(image_path)
        #     if crop_image_path!=None:
        #         image_list.append([image_path, label])

        for idx in configs.cropped_image_dir.glob('*.bmp'):
            print(idx)
            img_path = str(idx)
            img_label = img_path.split('/')[-1].split('-')[1]
            if img_label == 's':
                img_label = img_path.split('/')[-1].split('-')[2]
            image_list.append([img_path, img_label])
        
        crop_image_df = pd.DataFrame(image_list, columns=['image_path', 'label'])
        df_path = configs.root_dir / 'crop_image_paths.csv'
        crop_image_df.to_csv(df_path, index=False)

if __name__ == "__main__":

    if not os.path.exists(str(configs.augmented_image_dir)):
        os.makedirs(str(configs.augmented_image_dir))
    
    if not os.path.exists(str(configs.cropped_image_dir)):
        os.makedirs(str(configs.cropped_image_dir))

    image_df = pd.read_csv('../image_path.csv')
    augmentations = {
        'angles':[cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE, None],
        'flip':[None, 0 ,1]
    }
    aci = augment_crop_image(image_df, augmentations)
    aci.augment_image()
    aci.crop_image_dir()
