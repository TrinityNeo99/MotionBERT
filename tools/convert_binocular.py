import os
import sys
import pickle
import numpy as np
import random

sys.path.insert(0, os.getcwd())
sys.path.append("../")
print(os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_binocular import DataReaderBinocular
from tqdm import tqdm


def save_clips(subset_name, root_path, data_lefts, data_rights, data_labels, data_depth):
    assert len(data_lefts) == len(data_rights)
    len_data = len(data_lefts)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_data)):
        data_left, data_right, data_label = data_lefts[i], data_rights[i], data_labels[i]
        data_dict = {
            "data_left": data_left,
            "data_right": data_right,
            "data_label": data_label,
            "data_depth": data_depth
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict, myprofile)


datareader = DataReaderBinocular(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
                                 dt_file="coco_17_binocular.pkl",
                                 dt_root='../../dataset/binocular_data/sports')
                                 # dt_root="/mnt/weijiangning-pose-estimation-data/dual_camera_data")

# datareader = DataReaderBinocular(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243,
#                             dt_file="pingpong_v1_coco_17.pkl",
#                             # dt_file="h36m_sh_conf_cam_source_final.pkl",
#                             dt_root='/root/wjn/home/2024-human-pose-estimation-tutorial/MotionBERT/data/motion3d')
train_data_left, train_data_right, test_data_left, test_data_right, train_labels, test_labels, train_depth, test_depth = datareader.get_sliced_data()
print(train_data_left.shape, test_data_left.shape)

# temp model
train_labels = train_labels[:-1, :, :, :]
assert len(train_data_left) == len(train_labels)
assert len(test_data_left) == len(test_labels)

root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/front"
root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/behind"
root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/behind_cross_action"
root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/cross_view"
root_path = "../../dataset/binocular_data/pingpong_v1/binocular_pingpong_f243s81/binocular"
# root_path = "../../dataset/binocular_data/sports/binocular_f243s81/binocular"
root_path = "/mnt/weijiangning-pose-estimation-data/dual_camera_data/sports/binocular_f243s81/binocular"
# root_path = "/root/wjn/home/2024-human-pose-estimation-tutorial/MotionBERT/data/motion3d"
root_path = "../../dataset/binocular_data/sports/binocular_f243s81/binocular"
if not os.path.exists(root_path):
    os.makedirs(root_path)

save_clips("train", root_path, train_data_left, train_data_right, train_labels, train_depth)
save_clips("test", root_path, test_data_left, test_data_right, test_labels, test_depth)
