import scipy.io as scio
import os

data_path = 'D:/EGG_dataset/SEED-IV/eeg_feature_smooth/1'

total_all = 0
for subj_id in range(1, 16):
    files = [f for f in os.listdir(data_path)
             if f.split('_')[0] == str(subj_id) and f.endswith('.mat')]
    mat = scio.loadmat(os.path.join(data_path, files[0]))
    total = sum(mat[f'de_LDS{t}'].shape[1] for t in range(1, 25))
    total_all += total
    print(f"受试者 {subj_id:2d}：{total} 条样本")

print(f"\n所有受试者平均样本数：{total_all/15:.0f}")