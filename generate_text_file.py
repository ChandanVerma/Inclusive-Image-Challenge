import os

data_dir = '/home/osboxes/Desktop/kaggle/Inclusive_image_challenge/darknet/test_images/'

with open('openimages.txt', 'a+') as f:
    for img in os.listdir(data_dir):
        path = os.path.join(data_dir, img)
        f.write(path +'\n')
