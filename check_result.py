import scipy.io as scio
import numpy as np

# 加载保存的结果
result = scio.loadmat('D:\\EGG\\PR-PL\\result_SEED_IV_session_3.mat')
best_acc_mat = result['best_acc_mat'].flatten()

print("========== 最终结果 ==========")
print("每个受试者最高准确率：")
for i, acc in enumerate(best_acc_mat):
    print(f"  受试者 {i+1:2d}: {acc*100:.2f}%")

print(f"\n平均准确率：{np.mean(best_acc_mat)*100:.2f}%")
print(f"标准差：    {np.std(best_acc_mat)*100:.2f}%")
print(f"最终结果：  {np.mean(best_acc_mat)*100:.2f}% ± {np.std(best_acc_mat)*100:.2f}%")
print(f"\n论文结果：  81.32% ± 8.53%")
print(f"差距：      {abs(np.mean(best_acc_mat)*100 - 81.32):.2f}%")