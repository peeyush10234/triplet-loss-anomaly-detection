import glob 
import pandas as pd
import pathlib

root_dir = pathlib.Path.cwd().parent
image_sub_dir = root_dir / 'images/20201028'
image_sub_dir2_OK = root_dir / 'images/20201207/OK'
image_sub_dir2_NG = root_dir / 'images/20201207/NG'
image_sub_dir3_OK = root_dir / 'images/20210126/ok'
image_sub_dir3_NG = root_dir / 'images/20210126/ng'
print(root_dir)
image_path = []

for path in image_sub_dir.glob('*.bmp'):
    relative_path = ('/').join(list(path.parts)[-3:])
    path = str(path)
    image_name = path.split('/')[-1].split('.')[0]
    label = 'OK' if image_name[0]=='o' else 'NG'
    image_path.append([relative_path, label])

for path in image_sub_dir2_OK.glob('*.bmp'):
    relative_path = ('/').join(list(path.parts)[-4:])
    path = str(path)
    image_path.append([relative_path, 'OK'])

for path in image_sub_dir2_NG.glob('*.bmp'):
    relative_path = ('/').join(list(path.parts)[-4:])
    path = str(path)
    if path == '/Users/ptaneja/Documents/triplet-loss-anomaly-detection/images/20201207/NG/3説明.bmp':
        continue
    image_path.append([relative_path, 'NG'])

for path in image_sub_dir3_OK.glob('*.bmp'):
    relative_path = ('/').join(list(path.parts)[-4:])
    path = str(path)
    image_path.append([relative_path, 'OK'])

for path in image_sub_dir3_NG.glob('*.bmp'):
    relative_path = ('/').join(list(path.parts)[-4:])
    path = str(path)
    image_path.append([relative_path, 'NG'])

image_df = pd.DataFrame(image_path, columns=['image_path', 'label'])

image_df.to_csv('../image_path.csv', index=False)
# for path in glob.glob()