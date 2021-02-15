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
    
    def image_resize(self, image, width = 256, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized
    
    def augment_image(self):
        angles = self.augmentations['angles']
        flips = self.augmentations['flip']
        cropped_df = pd.read_csv(configs.root_dir / 'crop_image_paths.csv')
        image_list = []

        for path in configs.cropped_image_dir.glob('*.bmp'):
            image_name = str(path).split('/')[-1]
            image_path = path
            img = cv2.imread(str(image_path))
            ct=0
            label = ''
            for _ in image_name.split('-'):
                if 'o' in _ or 'O' in _:
                    label = 'OK'
                    break
            if label == '':
                label = 'NG'
            for angle in angles:
                for flip in flips:
                    img_aug = cv2.rotate(img, angle) if angle!=None else img
                    img_aug = cv2.flip(img_aug, flip) if flip!=None else img_aug
                    aug_img_name = f'{image_name}-{ct}.bmp'
                    aug_image_path = configs.augmented_image_dir / aug_img_name
                    img_aug = self.image_resize(img_aug)
                    plt.imsave(aug_image_path, img_aug)
                    image_list.append([aug_image_path, label, image_name])
                    ct+=1
        print(len(image_list))
        aug_df = pd.DataFrame(image_list, columns = ['image_path', 'label', 'img_name'])
        aug_df_path = configs.root_dir / 'augmented_images.csv'
        aug_df.to_csv(aug_df_path, index=False)
    
    def crop_save_image(self, file_path, index):
        img = cv2.imread(file_path)
        filename = str(index) + '-' + file_path.split('/')[-2] + '-' + file_path.split('/')[-1].split('.')[0]
        output = img.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT ,1,2000)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x,y,r) in circles:
                crop_img = output[y-(r+50):y+(r+50), x-(r+50):x+(r+50)]
                crop_image_path = f'{configs.cropped_image_dir}/{filename}.bmp'
                plt.imsave(crop_image_path,crop_img)
                return crop_image_path, filename
        return None, filename

    def crop_image_dir(self):
        # augment_image_dir = configs.augmented_image_dir
        image_list = []
        for index, row in self.df.iterrows():
            image_path = configs.root_dir / row['image_path']
            image_path, filename = self.crop_save_image(str(image_path), index)
            image_list.append([row['image_path'], filename])
        # for image_path in augment_image_dir.glob('*.bmp'):
        #     image_path = str(image_path)
        #     label = image_path.split('/')[-1].split('-')[1]
        #     if label == 's':
        #         label = image_path.split('/')[-1].split('-')[2]
        #     crop_image_path = self.crop_save_image(image_path)
        #     if crop_image_path!=None:
        #         image_list.append([image_path, label])

        # for idx in configs.cropped_image_dir.glob('*.bmp'):
        #     print(idx)
        #     img_path = str(idx)
        #     img_label = img_path.split('/')[-1].split('-')[1]
        #     if img_label == 's':
        #         img_label = img_path.split('/')[-1].split('-')[2]
        #     image_list.append([img_path, img_label])
        
        crop_image_df = pd.DataFrame(image_list, columns=['image_path', 'filename'])
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
    # aci.crop_image_dir()
    aci.augment_image()
