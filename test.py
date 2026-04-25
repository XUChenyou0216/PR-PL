import scipy.io as scio

# 改成你本地的实际路径
mat = scio.loadmat('D:/EGG_dataset/SEED-IV/eeg_feature_smooth/1/1_20160518.mat')

print("所有字段：")
print(mat.keys())

# 过滤掉系统字段，只看数据字段
keys = [k for k in mat.keys() if not k.startswith('__')]
print("\n数据字段：")
print(keys)

# 看第一个字段的形状
print("\n第一个字段的形状：")
print(mat[keys[0]].shape)