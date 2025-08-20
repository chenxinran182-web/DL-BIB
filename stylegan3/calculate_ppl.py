import pickle
import torch
import lpips
from tqdm import tqdm
import sys

# 添加 stylegan3 目录到 Python 路径
sys.path.append('/root/stylegan3')  # 替换为您的 stylegan3 目录路径

# 加载训练好的模型
weight_path = '/root/stylegan3-main/training-runs/00007/network-snapshot-000028.pkl'
with open(weight_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # 生成器移动到GPU

# 初始化 LPIPS 模型（用于计算图像感知距离）
loss_fn = lpips.LPIPS(net='vgg').cuda()

# 参数设置
num_samples = 10000  # 采样数量，建议10k-50k
epsilon = 1e-4  # 插值步长

# 计算 PPL
total_ppl = 0.0
for _ in tqdm(range(num_samples)):
    # 随机采样两个潜在向量
    z1 = torch.randn([1, G.z_dim]).cuda()
    z2 = torch.randn([1, G.z_dim]).cuda()
    
    # 插值
    t = torch.rand([]).cuda()
    z_int = z1 + (z2 - z1) * t
    z_int_e = z1 + (z2 - z1) * (t + epsilon)
    
    # 生成图像
    img1 = G(z_int, None)
    img2 = G(z_int_e, None)
    
    # 计算 LPIPS 距离
    with torch.no_grad():
        dist = loss_fn(img1, img2).mean()
    total_ppl += dist / (epsilon ** 2)

# 计算平均 PPL
ppl = total_ppl / num_samples
print(f'PPL: {ppl.item():.4f}')