import scipy.io as scio

mat = scio.loadmat('D:/EGG_dataset/SEED/feature/sub_1_session_1.mat')

data = mat['dataset_session1']
print(type(data))
print(data.shape)

# 尝试几种不同的索引方式
try:
    print("方式1:", data['feature'][0,0].shape)
except Exception as e:
    print("方式1失败:", e)

try:
    print("方式2:", data[0,0]['feature'].shape)
except Exception as e:
    print("方式2失败:", e)

try:
    print("方式3:", data[0]['feature'].shape)
except Exception as e:
    print("方式3失败:", e)

mat = scio.loadmat('D:/EGG_dataset/SEED/feature/sub_1_session_1.mat')
label = mat['dataset_session1']['label'][0,0]
print(label[:10])       # 看前10个值
print(set(label.flatten()))  # 看有哪些唯一值