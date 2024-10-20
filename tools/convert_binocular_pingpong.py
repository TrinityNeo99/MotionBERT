import os
import sys
import pickle
import numpy as np
import random

sys.path.insert(0, os.getcwd())
sys.path.append("../")
print(os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_binocular_pingpong import DataReaderBinocular
from tqdm import tqdm


def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict, myprofile)

datareader = DataReaderBinocular(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
                            dt_file="pingpong_v1_coco_17_cross_view.pkl",
                            dt_root='../../dataset/binocular_data/pingpong_v1')

# datareader = DataReaderBinocular(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
#                             dt_file="pingpong_v1_coco_17.pkl",
#                             # dt_file="h36m_sh_conf_cam_source_final.pkl",
#                             dt_root='/root/wjn/home/2024-human-pose-estimation-tutorial/MotionBERT/data/motion3d')
train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
print(train_data.shape, test_data.shape)
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/front"
root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/behind"
root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/behind_cross_action"
root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/cross_view"
# root_path = "/root/wjn/home/2024-human-pose-estimation-tutorial/MotionBERT/data/motion3d"
if not os.path.exists(root_path):
    os.makedirs(root_path)

save_clips("train", root_path, train_data, train_labels)
save_clips("test", root_path, test_data, test_labels)
