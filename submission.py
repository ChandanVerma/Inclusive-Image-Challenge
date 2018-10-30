from darknet import *
import tqdm
import pandas as pd
threshold = 0.2
cfg_dir = '/home/osboxes/Desktop/kaggle/Inclusive_image_challenge/darknet/cfg/'
backup_dir = '/home/osboxes/Desktop/kaggle/Inclusive_image_challenge/darknet/backup/'
metadata_dir = '/home/osboxes/Desktop/kaggle/Inclusive_image_challenge/darknet/'
data_dir = '/home/osboxes/Desktop/kaggle/Inclusive_image_challenge/darknet/data/'
data_extention_file_path = os.path.join(data_dir, 'openimages.data')
submit_file_path = "submission.csv"
cfg_path = os.path.join(cfg_dir, "yolov3-openimages.cfg")
weight_path = os.path.join(backup_dir, "yolov3-openimages.weights")
test_img_list_path = os.path.join(metadata_dir, "openimages.list")

gpu_index = 0
net = load_net(cfg_path.encode(),
               weight_path.encode(), 
               gpu_index)
meta = load_meta(data_extention_file_path.encode())

submit_dict = {"image_id": [], "labels": []}

with open(test_img_list_path, "r") as test_img_list_f:
    # tqdm run up to 1000(The # of test set)
    for line in tqdm.tqdm(test_img_list_f):
        image_id = line.strip().split('/')[-1].strip().split('.')[0]
        print(image_id)
        
        infer_result = detect(net, meta, line.strip().encode(), thresh=threshold)
        #print(infer_result)

        submit_line = ""
        for e in infer_result:
            label = e[0]
            print(label)
            #confi = e[1]
            #w = e[2][2]
            #h = e[2][3]
            #x = e[2][0]-w/2
            #y = e[2][1]-h/2
            #submit_line += "{} {} {} {} {} ".format(confi, x, y, w, h)
            submit_line += "{} ".format(label)

        submit_dict["image_id"].append(image_id)
        submit_dict["labels"].append(submit_line)

pd.DataFrame(submit_dict).to_csv(submit_file_path, index=False)

