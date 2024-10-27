"""
@Project: 2024-human-pose-estimation-tutorial
@FileName: dataset_motion_3d_mix.py
@Description: 自动描述，请及时修改
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/10/27 20:44 at PyCharm
"""
from torch.utils.data import Dataset
class CombinedDataset(Dataset):
    def __init__(self, dataset_1, dataset_2, ratio=3):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.ratio = ratio

    def __len__(self):
        return len(self.dataset_1) + (len(self.dataset_1) // self.ratio)

    def __getitem__(self, idx):
        if idx % (self.ratio + 1) < self.ratio:
            return self.dataset_1[idx - idx // (self.ratio + 1)], "dataset1"
        else:
            return self.dataset_2[(idx // (self.ratio + 1)) % len(self.dataset_2)], "dataset2"


def test_a():
    return 1,2,3

def test_b():
    return test_a(), "dataset1"

if __name__ == '__main__':
    dataset1 = [0,1,2,3,4,5]
    dataset2 = [6,7,8]
    c = CombinedDataset(dataset1, dataset2)
    for i in range(len(c)):
        print(c[i])
    print(test_b())