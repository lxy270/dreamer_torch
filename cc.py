import os
import numpy as np

# def count_npz_entries(directory):
#     total_entries = 0
#     # 遍历目录中的所有文件
#     for filename in os.listdir(directory):
#         total_entries+=1
#     print(total_entries)

#     return total_entries

# # 使用示例，替换 'your_directory_path' 为实际目录路径
# directory_path = '/home/chenjiehao/projects/dreamerv3-torch_ver/logdir/dmc_cheetah_run/train_eps'
# total_entries = count_npz_entries(directory_path)
# print(f"Total number of entries in all .npz files: {total_entries}")
data=np.load('/home/chenjiehao/projects/dreamerv3-torch_ver/logdir/dmc_cheetah_run/eval_eps/20240929T223729-44212d3b5a90425d95a0e1422f75b35a-1001.npz')
print(data['is_terminal'].shape)

