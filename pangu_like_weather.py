#!/usr/bin/env python3  # 指定由系统环境中的 python3 解释器运行本文件
"""
Pangu-like Weather Model — research-grade end-to-end skeleton in PyTorch
===========================================================================

This file gives you a *complete, runnable* pipeline that mirrors Pangu-Weather's
high-level logic while staying compact enough to iterate locally:

- 3D Earth-specific Transformer-style backbone over (level, lat, lon)
- 3D patch embedding + sinusoidal 3D positional encoding
- Circular padding in longitude to respect Earth's periodic boundary
- Train/val loops (mixed precision), gradient clipping, checkpointing
- Iterative inference for multi-step forecasts (e.g., +6h, +24h, ...)
- Dataset abstraction for ERA5-like NetCDF stacks via xarray (optional)

⚠️ Notes
- This is **not** the official Huawei Pangu-Weather implementation, but an
  educational, engineering-quality scaffold to help you stand up an end-to-end
  system. You can swap in your own variables, levels, and preprocessing.
- To match Pangu-Weather accuracy you will need large-scale training,
  careful variable standardization, and potentially architectural tweaks
  (hierarchical temporal aggregation, deeper transformer, etc.).

Usage (examples)
----------------
# 1) Training on your prepared dataset
python pangu_like_weather.py \
  --mode train \
  --data_root /data/era5_npz \
  --var_list t2m,msl,u10,v10,z,t,q,u,v \
  --levels 50,100,150,200,250,300,400,500,700,850,925,1000 \
  --in_hours 0 --out_hours 6 \
  --batch_size 2 --epochs 50 --accum_steps 2 --lr 3e-4

# 2) Inference — iterative 4×6h = 24h
python pangu_like_weather.py \
  --mode infer \
  --ckpt ./checkpoints/last.pt \
  --input_npz /data/samples/init_2025-01-01_00.npz \
  --steps 4 --step_hours 6 \
  --save ./pred_24h.npz

Data format (minimal NPZ)
-------------------------
We use simple .npz files for speed-of-iteration:
- Each sample contains an array `x` of shape [C, L, H, W], where
  C = number of variables (surface + upper-air concatenated),
  L = vertical levels, H = latitude points, W = longitude points.
- For training, the dataset expects pairs (x_t, x_t+Δt) with the *same* shape.

You can also adapt the XarrayDataset to read NetCDF/GRIB and assemble the
[C,L,H,W] tensors.

"""  # 顶部文档字符串：说明本文件目的、用法与数据格式
from __future__ import annotations  # 兼容旧版 Python 的注解前置解析（避免循环引用问题）
import os  # 操作文件和路径
import math  # 数学函数（如对数、指数等）
import json  # 预留：如需记录配置到 JSON
import time  # 计时与日志
import argparse  # 命令行参数解析
from pathlib import Path  # 路径对象化
from typing import List, Tuple, Optional  # 类型注解

import numpy as np  # 数组运算

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数（激活、loss 等）
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练工具
from torch.utils.data import Dataset, DataLoader  # 数据集与数据加载器

# =============================
# Utils（通用工具）
# =============================

def seed_everything(seed: int = 42):  # 设定随机种子，提升可复现性
    import random  # Python 内置随机库
    import numpy as np  # 再次导入以限定作用域（函数内部使用）
    import torch  # 同上
    random.seed(seed)  # 设定 Python 随机数种子
    np.random.seed(seed)  # 设定 NumPy 随机种子
    torch.manual_seed(seed)  # 设定 PyTorch CPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 设定所有 GPU 的随机种子
    torch.backends.cudnn.deterministic = False  # 不强制确定性（提升性能）
    torch.backends.cudnn.benchmark = True  # 允许 cuDNN 自动寻找最优算法


def count_parameters(model: nn.Module) -> int:  # 统计可训练参数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 遍历参数并求和


class CircularPad2d(nn.Module):  # 自定义 2D 环形（经度）padding 层
    """Circular padding along longitude (W) and optional lat (H).
    Defaults to circular on W only (Earth periodicity in longitude).
    """  # 说明：默认只对经度方向做循环填充，地球经度是周期边界
    def __init__(self, pad_w: int, pad_h: int = 0, circ_h: bool = False):  # 构造函数，设置经度/纬度 padding 量
        super().__init__()  # 调用父类构造
        self.pad_w = pad_w  # 记录经度方向需要填充的宽度
        self.pad_h = pad_h  # 记录纬度方向需要填充的高度
        self.circ_h = circ_h  # 是否对纬度方向也采用环形填充

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播定义
        # x: [B, C, L, H, W]  # 输入为五维张量（批次、通道、层、纬度、经度）
        if self.pad_w > 0:  # 若需要对经度方向做 padding
            left = x[..., -self.pad_w:]  # 取最右侧 pad_w 列作为左侧填充
            right = x[..., :self.pad_w]  # 取最左侧 pad_w 列作为右侧填充
            x = torch.cat([left, x, right], dim=-1)  # 按经度维拼接形成环形
        if self.pad_h > 0:  # 若需要对纬度方向做 padding
            if self.circ_h:  # 若选择纬度也采用环形
                top = x[..., -self.pad_h:, :]  # 取最下侧 pad_h 行作为顶部填充
                bottom = x[..., :self.pad_h, :]  # 取最上侧 pad_h 行作为底部填充
                x = torch.cat([top, x, bottom], dim=-2)  # 按纬度维拼接
            else:
                x = F.pad(x, (0,0, self.pad_h, self.pad_h, 0,0, 0,0, 0,0), mode='replicate')  # 非环形时用复制边缘的方式 pad
        return x  # 返回填充后的张量


# =============================
# Positional Encoding (3D sinusoidal)
# =============================

class SinusoidalPositionalEncoding3D(nn.Module):  # 三维正弦/余弦位置编码（用于 L/H/W 三轴）
    """3D sine-cosine positional encoding for (L, H, W) axes.
    Output shape matches token dim (D).
    """  # 注：实现为预计算表，按需添加到特征上
    def __init__(self, d_model: int, max_l: int, max_h: int, max_w: int):  # 指定特征维度与尺寸上限
        super().__init__()  # 调父类构造
        self.d_model = d_model  # 保存模型维度
        pe = self._build_pe(d_model, max_l, max_h, max_w)  # 构建位置编码表 [L,H,W,D]
        self.register_buffer('pe', pe, persistent=False)  # 注册为 buffer（不参与梯度/优化器）

    @staticmethod  # 静态方法，不依赖实例状态
    def _positional_encoding(d_half: int, length: int) -> torch.Tensor:  # 生成一维正弦/余弦位置编码
        position = torch.arange(length).unsqueeze(1)  # 位置索引列向量 [length,1]
        div_term = torch.exp(torch.arange(0, d_half, 2) * (-math.log(10000.0) / d_half))  # 频率项
        pe = torch.zeros(length, d_half)  # 初始化编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位放 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位放 cos
        return pe  # 返回 [length, d_half]

    def _build_pe(self, d_model: int, L: int, H: int, W: int) -> torch.Tensor:  # 组装三轴位置编码
        assert d_model % 6 == 0, "d_model must be divisible by 6 for 3 axes sine+cos"  # 每轴需要成对的 sin/cos
        d_each = d_model // 3  # 每个轴分到的通道数（含 sin 和 cos）
        d_half = d_each  # 直接用 d_each（内部函数已分奇偶）
        pe_l = self._positional_encoding(d_half, L)  # 纵向层数编码 [L, d_each]
        pe_h = self._positional_encoding(d_half, H)  # 纬度编码 [H, d_each]
        pe_w = self._positional_encoding(d_half, W)  # 经度编码 [W, d_each]
        # Broadcast add（下面进行维度扩展便于广播）
        pe_l = pe_l[:, None, None, :]      # [L,1,1,d]
        pe_h = pe_h[None, :, None, :]      # [1,H,1,d]
        pe_w = pe_w[None, None, :, :]      # [1,1,W,d]
        pe = torch.cat([pe_l.expand(L,H,W,d_half),  # 沿三轴广播到同一网格尺寸
                        pe_h.expand(L,H,W,d_half),
                        pe_w.expand(L,H,W,d_half)], dim=-1)  # 最后按通道拼接 -> [L,H,W,3*d_half]
        return pe  # 返回三维位置编码表

    def forward(self, LHW_tokens: torch.Tensor) -> torch.Tensor:  # 前向接口（此处保留占位）
        # LHW_tokens: [B, N, D] where N = L*H*W  # 若需要，可将 self.pe 映射后加到 tokens 上
        # We add pe later in the model after reshaping tokens back to grid if needed.
        return LHW_tokens  # 当前实现不直接修改输入


# =============================
# Patch Embedding / Recovery (3D)
# =============================

class PatchEmbed3D(nn.Module):  # 3D patch 嵌入（Conv3d 下采样做分块编码）
    def __init__(self, in_ch: int, embed_dim: int, patch: Tuple[int,int,int]):  # 指定输入通道、嵌入维度与 patch 大小
        super().__init__()  # 调用父类构造
        self.patch = patch  # 保存 patch 尺寸 (pL, pH, pW)
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch, stride=patch)  # 3D 卷积实现分块与通道映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        # x: [B,C,L,H,W] -> [B, D, L',H',W']  # 通过 stride=patch 下采样得到嵌入网格
        x = self.proj(x)  # 执行 3D 卷积投影到嵌入空间
        return x  # 输出嵌入特征

class PatchRecover3D(nn.Module):  # 3D patch 还原（ConvTranspose3d 上采样重建）
    def __init__(self, out_ch: int, embed_dim: int, patch: Tuple[int,int,int]):  # 指定输出通道等
        super().__init__()  # 调父类构造
        self.patch = patch  # 保存 patch 尺寸
        self.proj = nn.ConvTranspose3d(embed_dim, out_ch, kernel_size=patch, stride=patch)  # 反卷积还原分辨率

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向
        # x: [B, D, L',H',W'] -> [B, C, L,H,W]  # 将嵌入网格恢复到原始网格
        x = self.proj(x)  # 执行转置卷积
        return x  # 输出重建后的张量


# =============================
# Transformer blocks on flattened 3D tokens
# =============================

class MLP(nn.Module):  # Transformer 内的前馈网络
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):  # 指定维度、隐藏层与 dropout
        super().__init__()  # 调用父类构造
        self.fc1 = nn.Linear(dim, hidden)  # 第一层全连接
        self.act = nn.GELU()  # GELU 激活
        self.fc2 = nn.Linear(hidden, dim)  # 投回原维度
        self.drop = nn.Dropout(drop)  # dropout 层
    def forward(self, x):  # 前向传播
        x = self.fc1(x)  # 线性映射到隐藏层
        x = self.act(x)  # 激活
        x = self.drop(x)  # dropout
        x = self.fc2(x)  # 回到输入维度
        x = self.drop(x)  # 再次 dropout
        return x  # 返回输出

class TransformerBlock(nn.Module):  # 标准 Transformer 块（LN + MHA + 残差 + MLP）
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):  # 指定维度、头数等
        super().__init__()  # 父类构造
        self.norm1 = nn.LayerNorm(dim)  # 第一处 LayerNorm
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)  # 多头自注意力（batch 第一维）
        self.drop_path = nn.Dropout(drop)  # 简化的 drop path（这里用 Dropout 表示）
        self.norm2 = nn.LayerNorm(dim)  # 第二处 LayerNorm
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop)  # 前馈网络，隐藏维 = dim*mlp_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向
        # x: [B, N, D]  # 序列长度 N，特征维 D
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])  # 残差连接 + 自注意力
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差连接 + MLP
        return x  # 返回块输出


# =============================
# Pangu-like 3D Transformer（主模型）
# =============================

class PanguLike3D(nn.Module):  # 端到端 3D Transformer 模型
    def __init__(self,
                 in_ch: int,  # 输入通道数（变量数）
                 out_ch: int,  # 输出通道数（一般与输入一致，用于直接预测未来场）
                 levels: int,  # 垂直层数 L
                 lat: int,  # 纬向格点数 H
                 lon: int,  # 经向格点数 W
                 embed_dim: int = 384,  # token 嵌入维度
                 depth: int = 8,  # Transformer 堆叠层数
                 heads: int = 8,  # 注意力头数
                 patch: Tuple[int,int,int] = (1,4,4),  # 3D patch 大小（L,H,W）
                 dropout: float = 0.0,  # dropout 比例
                 ): 
        super().__init__()  # 父类构造
        self.in_ch = in_ch  # 保存输入通道
        self.out_ch = out_ch  # 保存输出通道
        self.levels = levels  # 保存层数
        self.lat = lat  # 保存纬度点数
        self.lon = lon  # 保存经度点数
        self.patch = patch  # 保存 patch 尺寸
        self.embed = PatchEmbed3D(in_ch, embed_dim, patch)  # 3D patch 嵌入
        self.pos = None  # 位置编码（此处未直接使用，可扩展）
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio=4.0, drop=dropout)  # 构建若干 TransformerBlock
            for _ in range(depth)
        ])
        self.recover = PatchRecover3D(out_ch, embed_dim, patch)  # 3D patch 还原，映射回物理场

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        # x: [B,C,L,H,W]  # 输入为时刻 t 的三维气象场
        B, C, L, H, W = x.shape  # 解包形状
        x = self.embed(x)              # [B, D, L',H',W']  # 3D 卷积分块编码
        B, D, Lp, Hp, Wp = x.shape  # 记录下采样后的尺寸
        # flatten spatial
        x = x.permute(0,2,3,4,1).contiguous().view(B, Lp*Hp*Wp, D)  # [B, N, D]  # 将 3D 网格展平成 token 序列
        for blk in self.blocks:  # 依次通过 Transformer 层
            x = blk(x)  # 块前向
        # reshape back
        x = x.view(B, Lp, Hp, Wp, D).permute(0,4,1,2,3).contiguous()  # [B,D,L',H',W']  # 还原为嵌入网格
        x = self.recover(x)            # [B, out_ch, L,H,W]  # 反卷积恢复到原始分辨率与通道
        return x  # 输出为下一时刻（或 Δt 后）的预测场


# =============================
# Datasets（数据集）
# =============================

class NPZPairDataset(Dataset):  # 使用 .npz 文件对 (x,y) 训练的简易数据集
    """Pairs of (input, target) stored as .npz with keys 'x' and 'y'."""  # 约定 .npz 内含 x 与 y
    def __init__(self, root: str):  # root 为数据目录
        self.files = sorted(str(p) for p in Path(root).glob('*.npz'))  # 扫描目录下所有 .npz 文件
        if len(self.files) == 0:  # 若未找到文件
            raise FileNotFoundError(f"No .npz found in {root}")  # 抛出异常，提醒用户

    def __len__(self):  # 返回样本数量
        return len(self.files)  # 即文件个数

    def __getitem__(self, idx: int):  # 读取第 idx 个样本
        arr = np.load(self.files[idx])  # 读取 .npz
        x = arr['x'].astype(np.float32)  # [C,L,H,W]  # 输入张量
        y = arr['y'].astype(np.float32)  # [C,L,H,W]  # 目标张量（Δt 后）
        return torch.from_numpy(x), torch.from_numpy(y)  # 转成 torch 张量返回


class XarrayDataset(Dataset):  # 可选：从 NetCDF/GRIB 读取并组装 [C,L,H,W]
    """
    Optional: load from NetCDF/GRIB assembled into tensors.
    Provide two lists of paths of equal length (inputs, targets).
    Each file must be convertible to [C,L,H,W].
    """  # 文档：要求 inputs 与 targets 列表等长
    def __init__(self, inputs: List[str], targets: List[str], var_order: Optional[List[str]] = None):  # 可指定变量顺序
        try:
            import xarray as xr  # noqa: F401  # 尝试导入 xarray（仅在需要时）
        except Exception as e:
            raise ImportError("xarray is required for XarrayDataset. pip install xarray netcdf4 cfgrib") from e  # 缺失依赖提示
        assert len(inputs) == len(targets)  # 校验列表长度一致
        self.inputs = inputs  # 保存输入文件路径列表
        self.targets = targets  # 保存目标文件路径列表
        self.var_order = var_order  # 变量顺序（可选）

    def __len__(self):  # 返回样本数
        return len(self.inputs)  # 与 inputs 列表长度相同

    def _to_tensor(self, path: str) -> torch.Tensor:  # 将单个文件转为张量
        import xarray as xr  # 导入 xarray
        ds = xr.open_dataset(path)  # 打开数据集
        if self.var_order is None:  # 若未指定变量顺序
            data = np.stack([ds[v].values for v in ds.data_vars], axis=0)  # 按数据集默认变量顺序堆叠
        else:
            data = np.stack([ds[v].values for v in self.var_order], axis=0)  # 按指定顺序堆叠
        # Expect [C,L,H,W] or [C,H,W] -> add level dim if needed
        if data.ndim == 3:  # 若缺少层维度
            data = data[:, None, ...]  # 插入 L 维（大小为 1）以对齐形状
        return torch.from_numpy(data.astype(np.float32))  # 转为 float32 的 torch 张量

    def __getitem__(self, idx: int):  # 获取样本对
        x = self._to_tensor(self.inputs[idx])  # 读取输入
        y = self._to_tensor(self.targets[idx])  # 读取目标
        return x, y  # 返回张量对


# =============================
# Training / Inference（训练 / 推理）
# =============================

@dataclass_init = False  # 简单标记（无实际功能，仅说明不是 dataclass）
class TrainConfig:  # 训练配置容器（可替代 argparse 使用）
    def __init__(self, **kwargs):  # 可通过关键字参数覆盖默认设置
        self.batch_size = 2  # 批大小
        self.epochs = 20  # 训练轮次
        self.lr = 3e-4  # 学习率
        self.weight_decay = 1e-4  # 权重衰减
        self.accum_steps = 1  # 梯度累积步数
        self.clip_grad = 1.0  # 梯度裁剪阈值（范数）
        self.num_workers = 4  # DataLoader 并行读取线程
        self.mixed_precision = True  # 是否开启混合精度
        self.save_dir = './checkpoints'  # 模型保存目录
        for k,v in kwargs.items():  # 遍历传入的配置覆盖默认值
            setattr(self, k, v)  # 动态设置属性


def make_model_from_sample(sample: torch.Tensor, embed_dim=384, depth=8, heads=8, patch=(1,4,4)) -> PanguLike3D:  # 基于样本形状构建模型
    # sample: [C,L,H,W]  # 使用样本推断通道/空间尺寸
    C, L, H, W = sample.shape  # 解包形状
    model = PanguLike3D(
        in_ch=C,  # 输入通道数
        out_ch=C,  # 输出通道数（与输入一致，做自回归式预测）
        levels=L,  # 垂直层数
        lat=H,  # 纬向格点数
        lon=W,  # 经向格点数
        embed_dim=embed_dim,  # 嵌入维度
        depth=depth,  # Transformer 层数
        heads=heads,  # 注意力头数
        patch=patch,  # 3D patch 大小
    )
    return model  # 返回模型


def save_ckpt(path: str, model: nn.Module, optim: torch.optim.Optimizer, scaler: Optional[GradScaler], step: int):  # 保存检查点
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 确保目录存在
    torch.save({
        'model': model.state_dict(),  # 模型权重
        'optim': optim.state_dict(),  # 优化器状态
        'scaler': scaler.state_dict() if scaler is not None else None,  # 混合精度缩放器状态
        'step': step,  # 当前全局 step
    }, path)  # 序列化到文件


def load_ckpt(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None, scaler: Optional[GradScaler] = None) -> int:  # 加载检查点
    ck = torch.load(path, map_location='cpu')  # 从文件载入到 CPU
    model.load_state_dict(ck['model'])  # 恢复模型权重
    if optim is not None and 'optim' in ck:  # 若提供优化器且存在其状态
        optim.load_state_dict(ck['optim'])  # 恢复优化器
    if scaler is not None and ck.get('scaler') is not None:  # 若使用混合精度且有缩放器状态
        scaler.load_state_dict(ck['scaler'])  # 恢复缩放器
    return int(ck.get('step', 0))  # 返回保存的 step（默认 0）


def train_loop(args):  # 训练主循环
    seed_everything(42)  # 固定随机性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU/CPU

    # Build dataset
    train_ds = NPZPairDataset(args.data_root)  # 构建训练数据集（npz 对）
    val_ds = None  # 预留：可扩展验证集
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)  # 训练数据加载器

    # Build model from a probe sample
    x0, y0 = train_ds[0]  # 取一个样本以获知形状
    model = make_model_from_sample(x0, embed_dim=args.embed_dim, depth=args.depth, heads=args.heads, patch=tuple(args.patch))  # 构建模型
    model.to(device)  # 将模型移动到设备
    print(f"Model params: {count_parameters(model)/1e6:.2f} M")  # 打印参数量（百万级）

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW 优化器
    scaler = GradScaler(enabled=args.mixed_precision)  # AMP 缩放器
    start_step = 0  # 起始 step（预留）

    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    best_loss = float('inf')  # 记录最优损失

    step = 0  # 全局步数计数
    for epoch in range(args.epochs):  # 训练多个 epoch
        model.train()  # 切换到训练模式
        t0 = time.time()  # 记录起始时间
        running = 0.0  # 累计损失
        opt.zero_grad(set_to_none=True)  # 清空梯度
        for i, (x,y) in enumerate(train_loader):  # 遍历 mini-batch
            x = x.to(device)  # 输入送设备
            y = y.to(device)  # 目标送设备
            with autocast(enabled=args.mixed_precision):  # 开启混合精度上下文
                y_hat = model(x)  # 前向计算预测
                loss = F.mse_loss(y_hat, y)  # MSE 损失
            scaler.scale(loss / args.accum_steps).backward()  # 按累积步数缩放反传
            if (i+1) % args.accum_steps == 0:  # 达到累积步数则进行一次优化器 step
                if args.clip_grad is not None:  # 若启用梯度裁剪
                    scaler.unscale_grad_(opt)  # 反缩放以获得真实梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)  # 按范数裁剪
                scaler.step(opt)  # 优化器更新
                scaler.update()  # AMP 缩放器更新
                opt.zero_grad(set_to_none=True)  # 清空累计的梯度
                step += 1  # 全局步数 +1
            running += loss.item()  # 累加当前 batch 的标量损失

        epoch_loss = running / len(train_loader)  # 计算 epoch 平均损失
        dt = time.time() - t0  # 计算耗时
        print(f"Epoch {epoch+1}/{args.epochs} - train MSE: {epoch_loss:.6f} - {dt:.1f}s")  # 打印日志

        # Save last
        save_ckpt(os.path.join(args.save_dir, 'last.pt'), model, opt, scaler, step)  # 每个 epoch 保存 last.pt
        if epoch_loss < best_loss:  # 若取得更好结果
            best_loss = epoch_loss  # 更新最优
            save_ckpt(os.path.join(args.save_dir, 'best.pt'), model, opt, scaler, step)  # 保存 best.pt
            print(f"  ↳ improved, saved best.pt (MSE {best_loss:.6f})")  # 打印改进提示


@torch.no_grad()  # 禁用梯度（推理阶段）
def infer_loop(args):  # 推理主流程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # Load initial state [C,L,H,W]
    init = np.load(args.input_npz)['x'].astype(np.float32)  # 从 npz 读入初始场（键 x）
    x = torch.from_numpy(init)[None]  # [1,C,L,H,W]  # 扩展 batch 维度

    model = make_model_from_sample(x[0], embed_dim=args.embed_dim, depth=args.depth, heads=args.heads, patch=tuple(args.patch))  # 构建模型骨架
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])  # 加载权重
    model.to(device).eval()  # 移动到设备并切换到评估模式

    preds = []  # 用于收集多步预测
    cur = x.to(device)  # 当前输入初始化为初始场
    for s in range(args.steps):  # 循环多步（迭代）
        y_hat = model(cur)  # one step forecast  # 单步预测 Δt
        preds.append(y_hat[0].cpu().numpy())  # 收集当前步预测（去除 batch 维，搬回 CPU）
        # iterative: feed next
        cur = y_hat  # 自回归：将输出作为下一步的输入

    arr = np.stack(preds, axis=0)  # [S, C,L,H,W]  # 按时间步堆叠所有预测
    os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None  # 确保保存目录存在
    np.savez_compressed(args.save, y=arr)  # 保存为压缩 npz（键 y）
    print(f"Saved predictions: {args.save} (shape {arr.shape})")  # 日志打印保存信息


# =============================
# CLI（命令行接口）
# =============================

def build_argparser():  # 构建命令行参数解析器
    p = argparse.ArgumentParser(description='Pangu-like Weather Transformer')  # 创建解析器并添加描述
    p.add_argument('--mode', choices=['train','infer'], required=True)  # 运行模式：训练或推理
    p.add_argument('--data_root', type=str, help='folder of .npz pairs (train)')  # 训练数据目录
    p.add_argument('--input_npz', type=str, help='single .npz with key x (infer)')  # 推理输入文件
    p.add_argument('--ckpt', type=str, help='checkpoint path for inference or resume')  # 检查点路径
    p.add_argument('--save', type=str, default='./preds.npz')  # 推理输出保存路径

    # Model
    p.add_argument('--embed_dim', type=int, default=384)  # 模型嵌入维度
    p.add_argument('--depth', type=int, default=8)  # Transformer 层数
    p.add_argument('--heads', type=int, default=8)  # 注意力头数
    p.add_argument('--patch', type=int, nargs=3, default=(1,4,4))  # 3D patch 大小（L H W）

    # Train
    p.add_argument('--batch_size', type=int, default=2)  # 批大小
    p.add_argument('--epochs', type=int, default=20)  # 训练轮数
    p.add_argument('--lr', type=float, default=3e-4)  # 学习率
    p.add_argument('--weight_decay', type=float, default=1e-4)  # 权重衰减
    p.add_argument('--accum_steps', type=int, default=1)  # 梯度累积步数
    p.add_argument('--clip_grad', type=float, default=1.0)  # 梯度裁剪阈值
    p.add_argument('--num_workers', type=int, default=4)  # DataLoader 线程数
    p.add_argument('--mixed_precision', action='store_true')  # 是否启用混合精度（通过命令行开关）
    p.add_argument('--save_dir', type=str, default='./checkpoints')  # 模型保存目录

    # Inference
    p.add_argument('--steps', type=int, default=4)  # 推理步数（迭代次数）
    p.add_argument('--step_hours', type=int, default=6)  # 每步代表的小时数（元信息，供记录/对齐用）

    return p  # 返回解析器


def main():  # 程序入口
    parser = build_argparser()  # 获取解析器
    args = parser.parse_args()  # 解析命令行参数

    if args.mode == 'train':  # 若选择训练模式
        if not args.data_root:  # 校验训练数据目录是否提供
            raise ValueError('--data_root is required for training')  # 未提供则报错
        train_loop(args)  # 进入训练流程
    elif args.mode == 'infer':  # 若选择推理模式
        if not args.input_npz or not args.ckpt:  # 校验必需文件
            raise ValueError('--input_npz and --ckpt are required for inference')  # 未提供则报错
        infer_loop(args)  # 进入推理流程


if __name__ == '__main__':  # 脚本直接运行时生效（非作为模块导入）
    main()  # 调用主函数
#!/usr/bin/env python3  # 指定由系统环境中的 python3 解释器运行本文件
"""
Pangu-like Weather Model — research-grade end-to-end skeleton in PyTorch
===========================================================================

This file gives you a *complete, runnable* pipeline that mirrors Pangu-Weather's
high-level logic while staying compact enough to iterate locally:

- 3D Earth-specific Transformer-style backbone over (level, lat, lon)
- 3D patch embedding + sinusoidal 3D positional encoding
- Circular padding in longitude to respect Earth's periodic boundary
- Train/val loops (mixed precision), gradient clipping, checkpointing
- Iterative inference for multi-step forecasts (e.g., +6h, +24h, ...)
- Dataset abstraction for ERA5-like NetCDF stacks via xarray (optional)

⚠️ Notes
- This is **not** the official Huawei Pangu-Weather implementation, but an
  educational, engineering-quality scaffold to help you stand up an end-to-end
  system. You can swap in your own variables, levels, and preprocessing.
- To match Pangu-Weather accuracy you will need large-scale training,
  careful variable standardization, and potentially architectural tweaks
  (hierarchical temporal aggregation, deeper transformer, etc.).

Usage (examples)
----------------
# 1) Training on your prepared dataset
python pangu_like_weather.py \
  --mode train \
  --data_root /data/era5_npz \
  --var_list t2m,msl,u10,v10,z,t,q,u,v \
  --levels 50,100,150,200,250,300,400,500,700,850,925,1000 \
  --in_hours 0 --out_hours 6 \
  --batch_size 2 --epochs 50 --accum_steps 2 --lr 3e-4

# 2) Inference — iterative 4×6h = 24h
python pangu_like_weather.py \
  --mode infer \
  --ckpt ./checkpoints/last.pt \
  --input_npz /data/samples/init_2025-01-01_00.npz \
  --steps 4 --step_hours 6 \
  --save ./pred_24h.npz

Data format (minimal NPZ)
-------------------------
We use simple .npz files for speed-of-iteration:
- Each sample contains an array `x` of shape [C, L, H, W], where
  C = number of variables (surface + upper-air concatenated),
  L = vertical levels, H = latitude points, W = longitude points.
- For training, the dataset expects pairs (x_t, x_t+Δt) with the *same* shape.

You can also adapt the XarrayDataset to read NetCDF/GRIB and assemble the
[C,L,H,W] tensors.

"""  # 顶部文档字符串：说明本文件目的、用法与数据格式
from __future__ import annotations  # 兼容旧版 Python 的注解前置解析（避免循环引用问题）
import os  # 操作文件和路径
import math  # 数学函数（如对数、指数等）
import json  # 预留：如需记录配置到 JSON
import time  # 计时与日志
import argparse  # 命令行参数解析
from pathlib import Path  # 路径对象化
from typing import List, Tuple, Optional  # 类型注解

import numpy as np  # 数组运算

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数（激活、loss 等）
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练工具
from torch.utils.data import Dataset, DataLoader  # 数据集与数据加载器

# =============================
# Utils（通用工具）
# =============================

def seed_everything(seed: int = 42):  # 设定随机种子，提升可复现性
    import random  # Python 内置随机库
    import numpy as np  # 再次导入以限定作用域（函数内部使用）
    import torch  # 同上
    random.seed(seed)  # 设定 Python 随机数种子
    np.random.seed(seed)  # 设定 NumPy 随机种子
    torch.manual_seed(seed)  # 设定 PyTorch CPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 设定所有 GPU 的随机种子
    torch.backends.cudnn.deterministic = False  # 不强制确定性（提升性能）
    torch.backends.cudnn.benchmark = True  # 允许 cuDNN 自动寻找最优算法


def count_parameters(model: nn.Module) -> int:  # 统计可训练参数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 遍历参数并求和


class CircularPad2d(nn.Module):  # 自定义 2D 环形（经度）padding 层
    """Circular padding along longitude (W) and optional lat (H).
    Defaults to circular on W only (Earth periodicity in longitude).
    """  # 说明：默认只对经度方向做循环填充，地球经度是周期边界
    def __init__(self, pad_w: int, pad_h: int = 0, circ_h: bool = False):  # 构造函数，设置经度/纬度 padding 量
        super().__init__()  # 调用父类构造
        self.pad_w = pad_w  # 记录经度方向需要填充的宽度
        self.pad_h = pad_h  # 记录纬度方向需要填充的高度
        self.circ_h = circ_h  # 是否对纬度方向也采用环形填充

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播定义
        # x: [B, C, L, H, W]  # 输入为五维张量（批次、通道、层、纬度、经度）
        if self.pad_w > 0:  # 若需要对经度方向做 padding
            left = x[..., -self.pad_w:]  # 取最右侧 pad_w 列作为左侧填充
            right = x[..., :self.pad_w]  # 取最左侧 pad_w 列作为右侧填充
            x = torch.cat([left, x, right], dim=-1)  # 按经度维拼接形成环形
        if self.pad_h > 0:  # 若需要对纬度方向做 padding
            if self.circ_h:  # 若选择纬度也采用环形
                top = x[..., -self.pad_h:, :]  # 取最下侧 pad_h 行作为顶部填充
                bottom = x[..., :self.pad_h, :]  # 取最上侧 pad_h 行作为底部填充
                x = torch.cat([top, x, bottom], dim=-2)  # 按纬度维拼接
            else:
                x = F.pad(x, (0,0, self.pad_h, self.pad_h, 0,0, 0,0, 0,0), mode='replicate')  # 非环形时用复制边缘的方式 pad
        return x  # 返回填充后的张量


# =============================
# Positional Encoding (3D sinusoidal)
# =============================

class SinusoidalPositionalEncoding3D(nn.Module):  # 三维正弦/余弦位置编码（用于 L/H/W 三轴）
    """3D sine-cosine positional encoding for (L, H, W) axes.
    Output shape matches token dim (D).
    """  # 注：实现为预计算表，按需添加到特征上
    def __init__(self, d_model: int, max_l: int, max_h: int, max_w: int):  # 指定特征维度与尺寸上限
        super().__init__()  # 调父类构造
        self.d_model = d_model  # 保存模型维度
        pe = self._build_pe(d_model, max_l, max_h, max_w)  # 构建位置编码表 [L,H,W,D]
        self.register_buffer('pe', pe, persistent=False)  # 注册为 buffer（不参与梯度/优化器）

    @staticmethod  # 静态方法，不依赖实例状态
    def _positional_encoding(d_half: int, length: int) -> torch.Tensor:  # 生成一维正弦/余弦位置编码
        position = torch.arange(length).unsqueeze(1)  # 位置索引列向量 [length,1]
        div_term = torch.exp(torch.arange(0, d_half, 2) * (-math.log(10000.0) / d_half))  # 频率项
        pe = torch.zeros(length, d_half)  # 初始化编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位放 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位放 cos
        return pe  # 返回 [length, d_half]

    def _build_pe(self, d_model: int, L: int, H: int, W: int) -> torch.Tensor:  # 组装三轴位置编码
        assert d_model % 6 == 0, "d_model must be divisible by 6 for 3 axes sine+cos"  # 每轴需要成对的 sin/cos
        d_each = d_model // 3  # 每个轴分到的通道数（含 sin 和 cos）
        d_half = d_each  # 直接用 d_each（内部函数已分奇偶）
        pe_l = self._positional_encoding(d_half, L)  # 纵向层数编码 [L, d_each]
        pe_h = self._positional_encoding(d_half, H)  # 纬度编码 [H, d_each]
        pe_w = self._positional_encoding(d_half, W)  # 经度编码 [W, d_each]
        # Broadcast add（下面进行维度扩展便于广播）
        pe_l = pe_l[:, None, None, :]      # [L,1,1,d]
        pe_h = pe_h[None, :, None, :]      # [1,H,1,d]
        pe_w = pe_w[None, None, :, :]      # [1,1,W,d]
        pe = torch.cat([pe_l.expand(L,H,W,d_half),  # 沿三轴广播到同一网格尺寸
                        pe_h.expand(L,H,W,d_half),
                        pe_w.expand(L,H,W,d_half)], dim=-1)  # 最后按通道拼接 -> [L,H,W,3*d_half]
        return pe  # 返回三维位置编码表

    def forward(self, LHW_tokens: torch.Tensor) -> torch.Tensor:  # 前向接口（此处保留占位）
        # LHW_tokens: [B, N, D] where N = L*H*W  # 若需要，可将 self.pe 映射后加到 tokens 上
        # We add pe later in the model after reshaping tokens back to grid if needed.
        return LHW_tokens  # 当前实现不直接修改输入


# =============================
# Patch Embedding / Recovery (3D)
# =============================

class PatchEmbed3D(nn.Module):  # 3D patch 嵌入（Conv3d 下采样做分块编码）
    def __init__(self, in_ch: int, embed_dim: int, patch: Tuple[int,int,int]):  # 指定输入通道、嵌入维度与 patch 大小
        super().__init__()  # 调用父类构造
        self.patch = patch  # 保存 patch 尺寸 (pL, pH, pW)
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch, stride=patch)  # 3D 卷积实现分块与通道映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        # x: [B,C,L,H,W] -> [B, D, L',H',W']  # 通过 stride=patch 下采样得到嵌入网格
        x = self.proj(x)  # 执行 3D 卷积投影到嵌入空间
        return x  # 输出嵌入特征

class PatchRecover3D(nn.Module):  # 3D patch 还原（ConvTranspose3d 上采样重建）
    def __init__(self, out_ch: int, embed_dim: int, patch: Tuple[int,int,int]):  # 指定输出通道等
        super().__init__()  # 调父类构造
        self.patch = patch  # 保存 patch 尺寸
        self.proj = nn.ConvTranspose3d(embed_dim, out_ch, kernel_size=patch, stride=patch)  # 反卷积还原分辨率

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向
        # x: [B, D, L',H',W'] -> [B, C, L,H,W]  # 将嵌入网格恢复到原始网格
        x = self.proj(x)  # 执行转置卷积
        return x  # 输出重建后的张量


# =============================
# Transformer blocks on flattened 3D tokens
# =============================

class MLP(nn.Module):  # Transformer 内的前馈网络
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):  # 指定维度、隐藏层与 dropout
        super().__init__()  # 调用父类构造
        self.fc1 = nn.Linear(dim, hidden)  # 第一层全连接
        self.act = nn.GELU()  # GELU 激活
        self.fc2 = nn.Linear(hidden, dim)  # 投回原维度
        self.drop = nn.Dropout(drop)  # dropout 层
    def forward(self, x):  # 前向传播
        x = self.fc1(x)  # 线性映射到隐藏层
        x = self.act(x)  # 激活
        x = self.drop(x)  # dropout
        x = self.fc2(x)  # 回到输入维度
        x = self.drop(x)  # 再次 dropout
        return x  # 返回输出

class TransformerBlock(nn.Module):  # 标准 Transformer 块（LN + MHA + 残差 + MLP）
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):  # 指定维度、头数等
        super().__init__()  # 父类构造
        self.norm1 = nn.LayerNorm(dim)  # 第一处 LayerNorm
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)  # 多头自注意力（batch 第一维）
        self.drop_path = nn.Dropout(drop)  # 简化的 drop path（这里用 Dropout 表示）
        self.norm2 = nn.LayerNorm(dim)  # 第二处 LayerNorm
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop)  # 前馈网络，隐藏维 = dim*mlp_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向
        # x: [B, N, D]  # 序列长度 N，特征维 D
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])  # 残差连接 + 自注意力
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差连接 + MLP
        return x  # 返回块输出


# =============================
# Pangu-like 3D Transformer（主模型）
# =============================

class PanguLike3D(nn.Module):  # 端到端 3D Transformer 模型
    def __init__(self,
                 in_ch: int,  # 输入通道数（变量数）
                 out_ch: int,  # 输出通道数（一般与输入一致，用于直接预测未来场）
                 levels: int,  # 垂直层数 L
                 lat: int,  # 纬向格点数 H
                 lon: int,  # 经向格点数 W
                 embed_dim: int = 384,  # token 嵌入维度
                 depth: int = 8,  # Transformer 堆叠层数
                 heads: int = 8,  # 注意力头数
                 patch: Tuple[int,int,int] = (1,4,4),  # 3D patch 大小（L,H,W）
                 dropout: float = 0.0,  # dropout 比例
                 ): 
        super().__init__()  # 父类构造
        self.in_ch = in_ch  # 保存输入通道
        self.out_ch = out_ch  # 保存输出通道
        self.levels = levels  # 保存层数
        self.lat = lat  # 保存纬度点数
        self.lon = lon  # 保存经度点数
        self.patch = patch  # 保存 patch 尺寸
        self.embed = PatchEmbed3D(in_ch, embed_dim, patch)  # 3D patch 嵌入
        self.pos = None  # 位置编码（此处未直接使用，可扩展）
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio=4.0, drop=dropout)  # 构建若干 TransformerBlock
            for _ in range(depth)
        ])
        self.recover = PatchRecover3D(out_ch, embed_dim, patch)  # 3D patch 还原，映射回物理场

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        # x: [B,C,L,H,W]  # 输入为时刻 t 的三维气象场
        B, C, L, H, W = x.shape  # 解包形状
        x = self.embed(x)              # [B, D, L',H',W']  # 3D 卷积分块编码
        B, D, Lp, Hp, Wp = x.shape  # 记录下采样后的尺寸
        # flatten spatial
        x = x.permute(0,2,3,4,1).contiguous().view(B, Lp*Hp*Wp, D)  # [B, N, D]  # 将 3D 网格展平成 token 序列
        for blk in self.blocks:  # 依次通过 Transformer 层
            x = blk(x)  # 块前向
        # reshape back
        x = x.view(B, Lp, Hp, Wp, D).permute(0,4,1,2,3).contiguous()  # [B,D,L',H',W']  # 还原为嵌入网格
        x = self.recover(x)            # [B, out_ch, L,H,W]  # 反卷积恢复到原始分辨率与通道
        return x  # 输出为下一时刻（或 Δt 后）的预测场


# =============================
# Datasets（数据集）
# =============================

class NPZPairDataset(Dataset):  # 使用 .npz 文件对 (x,y) 训练的简易数据集
    """Pairs of (input, target) stored as .npz with keys 'x' and 'y'."""  # 约定 .npz 内含 x 与 y
    def __init__(self, root: str):  # root 为数据目录
        self.files = sorted(str(p) for p in Path(root).glob('*.npz'))  # 扫描目录下所有 .npz 文件
        if len(self.files) == 0:  # 若未找到文件
            raise FileNotFoundError(f"No .npz found in {root}")  # 抛出异常，提醒用户

    def __len__(self):  # 返回样本数量
        return len(self.files)  # 即文件个数

    def __getitem__(self, idx: int):  # 读取第 idx 个样本
        arr = np.load(self.files[idx])  # 读取 .npz
        x = arr['x'].astype(np.float32)  # [C,L,H,W]  # 输入张量
        y = arr['y'].astype(np.float32)  # [C,L,H,W]  # 目标张量（Δt 后）
        return torch.from_numpy(x), torch.from_numpy(y)  # 转成 torch 张量返回


class XarrayDataset(Dataset):  # 可选：从 NetCDF/GRIB 读取并组装 [C,L,H,W]
    """
    Optional: load from NetCDF/GRIB assembled into tensors.
    Provide two lists of paths of equal length (inputs, targets).
    Each file must be convertible to [C,L,H,W].
    """  # 文档：要求 inputs 与 targets 列表等长
    def __init__(self, inputs: List[str], targets: List[str], var_order: Optional[List[str]] = None):  # 可指定变量顺序
        try:
            import xarray as xr  # noqa: F401  # 尝试导入 xarray（仅在需要时）
        except Exception as e:
            raise ImportError("xarray is required for XarrayDataset. pip install xarray netcdf4 cfgrib") from e  # 缺失依赖提示
        assert len(inputs) == len(targets)  # 校验列表长度一致
        self.inputs = inputs  # 保存输入文件路径列表
        self.targets = targets  # 保存目标文件路径列表
        self.var_order = var_order  # 变量顺序（可选）

    def __len__(self):  # 返回样本数
        return len(self.inputs)  # 与 inputs 列表长度相同

    def _to_tensor(self, path: str) -> torch.Tensor:  # 将单个文件转为张量
        import xarray as xr  # 导入 xarray
        ds = xr.open_dataset(path)  # 打开数据集
        if self.var_order is None:  # 若未指定变量顺序
            data = np.stack([ds[v].values for v in ds.data_vars], axis=0)  # 按数据集默认变量顺序堆叠
        else:
            data = np.stack([ds[v].values for v in self.var_order], axis=0)  # 按指定顺序堆叠
        # Expect [C,L,H,W] or [C,H,W] -> add level dim if needed
        if data.ndim == 3:  # 若缺少层维度
            data = data[:, None, ...]  # 插入 L 维（大小为 1）以对齐形状
        return torch.from_numpy(data.astype(np.float32))  # 转为 float32 的 torch 张量

    def __getitem__(self, idx: int):  # 获取样本对
        x = self._to_tensor(self.inputs[idx])  # 读取输入
        y = self._to_tensor(self.targets[idx])  # 读取目标
        return x, y  # 返回张量对


# =============================
# Training / Inference（训练 / 推理）
# =============================

@dataclass_init = False  # 简单标记（无实际功能，仅说明不是 dataclass）
class TrainConfig:  # 训练配置容器（可替代 argparse 使用）
    def __init__(self, **kwargs):  # 可通过关键字参数覆盖默认设置
        self.batch_size = 2  # 批大小
        self.epochs = 20  # 训练轮次
        self.lr = 3e-4  # 学习率
        self.weight_decay = 1e-4  # 权重衰减
        self.accum_steps = 1  # 梯度累积步数
        self.clip_grad = 1.0  # 梯度裁剪阈值（范数）
        self.num_workers = 4  # DataLoader 并行读取线程
        self.mixed_precision = True  # 是否开启混合精度
        self.save_dir = './checkpoints'  # 模型保存目录
        for k,v in kwargs.items():  # 遍历传入的配置覆盖默认值
            setattr(self, k, v)  # 动态设置属性


def make_model_from_sample(sample: torch.Tensor, embed_dim=384, depth=8, heads=8, patch=(1,4,4)) -> PanguLike3D:  # 基于样本形状构建模型
    # sample: [C,L,H,W]  # 使用样本推断通道/空间尺寸
    C, L, H, W = sample.shape  # 解包形状
    model = PanguLike3D(
        in_ch=C,  # 输入通道数
        out_ch=C,  # 输出通道数（与输入一致，做自回归式预测）
        levels=L,  # 垂直层数
        lat=H,  # 纬向格点数
        lon=W,  # 经向格点数
        embed_dim=embed_dim,  # 嵌入维度
        depth=depth,  # Transformer 层数
        heads=heads,  # 注意力头数
        patch=patch,  # 3D patch 大小
    )
    return model  # 返回模型


def save_ckpt(path: str, model: nn.Module, optim: torch.optim.Optimizer, scaler: Optional[GradScaler], step: int):  # 保存检查点
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 确保目录存在
    torch.save({
        'model': model.state_dict(),  # 模型权重
        'optim': optim.state_dict(),  # 优化器状态
        'scaler': scaler.state_dict() if scaler is not None else None,  # 混合精度缩放器状态
        'step': step,  # 当前全局 step
    }, path)  # 序列化到文件


def load_ckpt(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None, scaler: Optional[GradScaler] = None) -> int:  # 加载检查点
    ck = torch.load(path, map_location='cpu')  # 从文件载入到 CPU
    model.load_state_dict(ck['model'])  # 恢复模型权重
    if optim is not None and 'optim' in ck:  # 若提供优化器且存在其状态
        optim.load_state_dict(ck['optim'])  # 恢复优化器
    if scaler is not None and ck.get('scaler') is not None:  # 若使用混合精度且有缩放器状态
        scaler.load_state_dict(ck['scaler'])  # 恢复缩放器
    return int(ck.get('step', 0))  # 返回保存的 step（默认 0）


def train_loop(args):  # 训练主循环
    seed_everything(42)  # 固定随机性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU/CPU

    # Build dataset
    train_ds = NPZPairDataset(args.data_root)  # 构建训练数据集（npz 对）
    val_ds = None  # 预留：可扩展验证集
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)  # 训练数据加载器

    # Build model from a probe sample
    x0, y0 = train_ds[0]  # 取一个样本以获知形状
    model = make_model_from_sample(x0, embed_dim=args.embed_dim, depth=args.depth, heads=args.heads, patch=tuple(args.patch))  # 构建模型
    model.to(device)  # 将模型移动到设备
    print(f"Model params: {count_parameters(model)/1e6:.2f} M")  # 打印参数量（百万级）

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW 优化器
    scaler = GradScaler(enabled=args.mixed_precision)  # AMP 缩放器
    start_step = 0  # 起始 step（预留）

    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    best_loss = float('inf')  # 记录最优损失

    step = 0  # 全局步数计数
    for epoch in range(args.epochs):  # 训练多个 epoch
        model.train()  # 切换到训练模式
        t0 = time.time()  # 记录起始时间
        running = 0.0  # 累计损失
        opt.zero_grad(set_to_none=True)  # 清空梯度
        for i, (x,y) in enumerate(train_loader):  # 遍历 mini-batch
            x = x.to(device)  # 输入送设备
            y = y.to(device)  # 目标送设备
            with autocast(enabled=args.mixed_precision):  # 开启混合精度上下文
                y_hat = model(x)  # 前向计算预测
                loss = F.mse_loss(y_hat, y)  # MSE 损失
            scaler.scale(loss / args.accum_steps).backward()  # 按累积步数缩放反传
            if (i+1) % args.accum_steps == 0:  # 达到累积步数则进行一次优化器 step
                if args.clip_grad is not None:  # 若启用梯度裁剪
                    scaler.unscale_grad_(opt)  # 反缩放以获得真实梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)  # 按范数裁剪
                scaler.step(opt)  # 优化器更新
                scaler.update()  # AMP 缩放器更新
                opt.zero_grad(set_to_none=True)  # 清空累计的梯度
                step += 1  # 全局步数 +1
            running += loss.item()  # 累加当前 batch 的标量损失

        epoch_loss = running / len(train_loader)  # 计算 epoch 平均损失
        dt = time.time() - t0  # 计算耗时
        print(f"Epoch {epoch+1}/{args.epochs} - train MSE: {epoch_loss:.6f} - {dt:.1f}s")  # 打印日志

        # Save last
        save_ckpt(os.path.join(args.save_dir, 'last.pt'), model, opt, scaler, step)  # 每个 epoch 保存 last.pt
        if epoch_loss < best_loss:  # 若取得更好结果
            best_loss = epoch_loss  # 更新最优
            save_ckpt(os.path.join(args.save_dir, 'best.pt'), model, opt, scaler, step)  # 保存 best.pt
            print(f"  ↳ improved, saved best.pt (MSE {best_loss:.6f})")  # 打印改进提示


@torch.no_grad()  # 禁用梯度（推理阶段）
def infer_loop(args):  # 推理主流程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # Load initial state [C,L,H,W]
    init = np.load(args.input_npz)['x'].astype(np.float32)  # 从 npz 读入初始场（键 x）
    x = torch.from_numpy(init)[None]  # [1,C,L,H,W]  # 扩展 batch 维度

    model = make_model_from_sample(x[0], embed_dim=args.embed_dim, depth=args.depth, heads=args.heads, patch=tuple(args.patch))  # 构建模型骨架
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])  # 加载权重
    model.to(device).eval()  # 移动到设备并切换到评估模式

    preds = []  # 用于收集多步预测
    cur = x.to(device)  # 当前输入初始化为初始场
    for s in range(args.steps):  # 循环多步（迭代）
        y_hat = model(cur)  # one step forecast  # 单步预测 Δt
        preds.append(y_hat[0].cpu().numpy())  # 收集当前步预测（去除 batch 维，搬回 CPU）
        # iterative: feed next
        cur = y_hat  # 自回归：将输出作为下一步的输入

    arr = np.stack(preds, axis=0)  # [S, C,L,H,W]  # 按时间步堆叠所有预测
    os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None  # 确保保存目录存在
    np.savez_compressed(args.save, y=arr)  # 保存为压缩 npz（键 y）
    print(f"Saved predictions: {args.save} (shape {arr.shape})")  # 日志打印保存信息


# =============================
# CLI（命令行接口）
# =============================

def build_argparser():  # 构建命令行参数解析器
    p = argparse.ArgumentParser(description='Pangu-like Weather Transformer')  # 创建解析器并添加描述
    p.add_argument('--mode', choices=['train','infer'], required=True)  # 运行模式：训练或推理
    p.add_argument('--data_root', type=str, help='folder of .npz pairs (train)')  # 训练数据目录
    p.add_argument('--input_npz', type=str, help='single .npz with key x (infer)')  # 推理输入文件
    p.add_argument('--ckpt', type=str, help='checkpoint path for inference or resume')  # 检查点路径
    p.add_argument('--save', type=str, default='./preds.npz')  # 推理输出保存路径

    # Model
    p.add_argument('--embed_dim', type=int, default=384)  # 模型嵌入维度
    p.add_argument('--depth', type=int, default=8)  # Transformer 层数
    p.add_argument('--heads', type=int, default=8)  # 注意力头数
    p.add_argument('--patch', type=int, nargs=3, default=(1,4,4))  # 3D patch 大小（L H W）

    # Train
    p.add_argument('--batch_size', type=int, default=2)  # 批大小
    p.add_argument('--epochs', type=int, default=20)  # 训练轮数
    p.add_argument('--lr', type=float, default=3e-4)  # 学习率
    p.add_argument('--weight_decay', type=float, default=1e-4)  # 权重衰减
    p.add_argument('--accum_steps', type=int, default=1)  # 梯度累积步数
    p.add_argument('--clip_grad', type=float, default=1.0)  # 梯度裁剪阈值
    p.add_argument('--num_workers', type=int, default=4)  # DataLoader 线程数
    p.add_argument('--mixed_precision', action='store_true')  # 是否启用混合精度（通过命令行开关）
    p.add_argument('--save_dir', type=str, default='./checkpoints')  # 模型保存目录

    # Inference
    p.add_argument('--steps', type=int, default=4)  # 推理步数（迭代次数）
    p.add_argument('--step_hours', type=int, default=6)  # 每步代表的小时数（元信息，供记录/对齐用）

    return p  # 返回解析器


def main():  # 程序入口
    parser = build_argparser()  # 获取解析器
    args = parser.parse_args()  # 解析命令行参数

    if args.mode == 'train':  # 若选择训练模式
        if not args.data_root:  # 校验训练数据目录是否提供
            raise ValueError('--data_root is required for training')  # 未提供则报错
        train_loop(args)  # 进入训练流程
    elif args.mode == 'infer':  # 若选择推理模式
        if not args.input_npz or not args.ckpt:  # 校验必需文件
            raise ValueError('--input_npz and --ckpt are required for inference')  # 未提供则报错
        infer_loop(args)  # 进入推理流程


if __name__ == '__main__':  # 脚本直接运行时生效（非作为模块导入）
    main()  # 调用主函数
