#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------
# Coordinate Attention (CVPR 2021)
# Ref paper:
#   - factorize 2D global pooling into two 1D pooling along H/W
#   - concat -> 1x1 conv + BN + non-linear -> split -> 1x1 conv -> sigmoid -> reweight
# ----------------------------
class HSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(x)


class CoordAtt(nn.Module):
    """
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # (B,C,1,W)

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)                 # (B,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)     # (B,C,H+W,1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)        # (B,mip,1,W)

        a_h = torch.sigmoid(self.conv_h(x_h))  # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(x_w))  # (B,C,1,W)

        out = identity * a_h * a_w
        return out


# ----------------------------
# Agent Attention (2D feature map)
# 贴近论文公式：
#   O = softmax(QA^T + B2) · softmax(AK^T + B1) · V + DWC(V)
# 这里的 Q/K/V 来自 feature map；A 来自 pooling；B1/B2 为 agent bias（可选）
# ----------------------------
class AgentAttention2D(nn.Module):
    """
    输入: (B, C, H, W)
    - A: agent tokens（默认通过 pooling 得到，A=a*a）
    - Step1 Agent Aggregation: A 作为 query，去聚合 token 的 V
    - Step2 Agent Broadcast: token 的 Q 去 query agent features
    - DWC: depthwise conv 作为多样性/局部路径
    - Agent Bias: 可选（B1/B2），用2D相对位置bias table实现（对小feature map很稳）
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        agent_tokens: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_agent_bias: bool = False,
        agent_bias_max_hw: int = 32,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 让 agent_tokens = a*a
        a = int(round(agent_tokens ** 0.5))
        if a * a != agent_tokens:
            a = int((agent_tokens ** 0.5) + 0.9999)
            agent_tokens = a * a
        self.agent_hw = a
        self.agent_tokens = agent_tokens

        # token projections
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # agent projections (Aq/Ak) from pooled tokens
        self.agent_q = nn.Linear(dim, dim, bias=False)
        self.agent_k = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        # DWC path (paper uses DWC(V); in CNN feature map上做 depthwise conv即可)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.norm = nn.LayerNorm(dim)

        # -------- Agent Bias（可选）--------
        self.use_agent_bias = use_agent_bias
        self.agent_bias_max_hw = int(agent_bias_max_hw)
        if self.use_agent_bias:
            # 每个 head 一个 2D 相对位置 bias table：
            # rel_y in [-(maxH-1), +(maxH-1)] => 2*maxH-1
            # rel_x in [-(maxW-1), +(maxW-1)] => 2*maxW-1
            size = (2 * self.agent_bias_max_hw - 1) * (2 * self.agent_bias_max_hw - 1)
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, size))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # cache：不同 (H,W,a) 的索引避免重复算
        self._bias_index_cache: Dict[str, torch.Tensor] = {}

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) -> (B, heads, N, head_dim)
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _get_agent_coords_on_token_grid(self, H: int, W: int, a: int, device) -> torch.Tensor:
        # agent 网格 (a,a) 映射到 token 网格 (H,W) 上的“代表位置”（整数）
        # 用线性映射：0..a-1 -> 0..H-1 / 0..W-1
        if a == 1:
            ys = torch.tensor([0], device=device, dtype=torch.long)
            xs = torch.tensor([0], device=device, dtype=torch.long)
        else:
            ys = torch.round(torch.linspace(0, H - 1, steps=a, device=device)).long()
            xs = torch.round(torch.linspace(0, W - 1, steps=a, device=device)).long()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # (A,2)
        return coords

    def _get_token_coords(self, H: int, W: int, device) -> torch.Tensor:
        ys = torch.arange(H, device=device, dtype=torch.long)
        xs = torch.arange(W, device=device, dtype=torch.long)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # (N,2)
        return coords

    def _build_bias(self, H: int, W: int, a: int, device) -> torch.Tensor:
        """
        返回 bias_B2: (1, heads, N, A)  用于 softmax(QA^T + B2)
        同时 B1 可以用 B2 的转置/取负相对位移得到，但这里直接用转置近似共享即可。
        """
        key = f"H{H}_W{W}_a{a}_max{self.agent_bias_max_hw}"
        if key in self._bias_index_cache:
            idx = self._bias_index_cache[key].to(device=device)
        else:
            token = self._get_token_coords(H, W, device)          # (N,2)
            agent = self._get_agent_coords_on_token_grid(H, W, a, device)  # (A,2)

            # rel = token - agent
            # token: (N,2) -> (N,1,2); agent: (A,2) -> (1,A,2)
            rel = token[:, None, :] - agent[None, :, :]  # (N,A,2)
            rel_y = rel[..., 0].clamp(-(self.agent_bias_max_hw - 1), self.agent_bias_max_hw - 1)
            rel_x = rel[..., 1].clamp(-(self.agent_bias_max_hw - 1), self.agent_bias_max_hw - 1)

            # map (rel_y, rel_x) -> flattened index
            # shift to [0, 2*max-2]
            rel_y = rel_y + (self.agent_bias_max_hw - 1)
            rel_x = rel_x + (self.agent_bias_max_hw - 1)
            idx = rel_y * (2 * self.agent_bias_max_hw - 1) + rel_x  # (N,A)
            idx = idx.reshape(-1).contiguous()  # (N*A,)
            # cache on cpu
            self._bias_index_cache[key] = idx.detach().cpu()

        # lookup
        # rel_pos_bias: (heads, L)
        bias = self.rel_pos_bias[:, idx]  # (heads, N*A)
        bias = bias.view(self.num_heads, H * W, a * a)  # (heads, N, A)
        return bias.unsqueeze(0)  # (1, heads, N, A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        a = self.agent_hw
        N = H * W

        # token projections
        q = self.q(x).flatten(2).transpose(1, 2)  # (B, N, C)
        k = self.k(x).flatten(2).transpose(1, 2)  # (B, N, C)
        v = self.v(x).flatten(2).transpose(1, 2)  # (B, N, C)

        # agents from pooled feature (pooling(Q) 在 CNN 上差别不大，这里直接 pooling(x))
        pooled = F.adaptive_avg_pool2d(x, (a, a))        # (B,C,a,a)
        agents = pooled.flatten(2).transpose(1, 2)       # (B,A,C)

        # agent q/k
        aq = self.agent_q(agents)  # (B,A,C)
        ak = self.agent_k(agents)  # (B,A,C)

        # heads
        qh = self._reshape_heads(q)    # (B,h,N,d)
        kh = self._reshape_heads(k)    # (B,h,N,d)
        vh = self._reshape_heads(v)    # (B,h,N,d)
        aqh = self._reshape_heads(aq)  # (B,h,A,d)
        akh = self._reshape_heads(ak)  # (B,h,A,d)

        # ------- Agent Bias（可选）-------
        if self.use_agent_bias:
            # B2: (1,h,N,A)
            B2 = self._build_bias(H, W, a, x.device)  # token->agent
            # B1: (1,h,A,N) 这里用转置近似共享（实际论文是分别的 B1/B2）
            B1 = B2.transpose(-2, -1).contiguous()
        else:
            B2 = None
            B1 = None

        # 1) Agent Aggregation: VA = softmax(AK^T + B1) · V
        attn_a = (aqh * self.scale) @ kh.transpose(-2, -1)  # (B,h,A,N)
        if B1 is not None:
            attn_a = attn_a + B1
        attn_a = attn_a.softmax(dim=-1)
        attn_a = self.attn_drop(attn_a)
        VA = attn_a @ vh  # (B,h,A,d)

        # 2) Agent Broadcast: O = softmax(QA^T + B2) · VA
        attn_b = (qh * self.scale) @ akh.transpose(-2, -1)  # (B,h,N,A)
        if B2 is not None:
            attn_b = attn_b + B2
        attn_b = attn_b.softmax(dim=-1)
        attn_b = self.attn_drop(attn_b)
        out = attn_b @ VA  # (B,h,N,d)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # (B,N,C)

        # proj + norm
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.norm(out)

        # back to (B,C,H,W)
        out = out.transpose(1, 2).contiguous().view(B, C, H, W)

        # DWC(V) / local path（论文强调的多样性补偿）
        local = self.dwconv(x)

        # residual
        return x + out + local


class ResNet18WithCoordAndAgentAttn(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        use_coord_attn: bool,
        coord_reduction: int,
        use_agent_attn: bool,
        agent_tokens: int,
        agent_heads: int,
        use_agent_bias: bool,
        agent_bias_max_hw: int,
        attn_order: str = "coord_then_agent",  # or "agent_then_coord"
    ):
        super().__init__()
        if pretrained:
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = models.resnet18(pretrained=True)
        else:
            backbone = models.resnet18(weights=None) if "weights" in models.resnet18.__code__.co_varnames else models.resnet18(pretrained=False)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.use_coord_attn = use_coord_attn
        self.use_agent_attn = use_agent_attn
        self.attn_order = attn_order

        self.coord_attn = CoordAtt(inp=512, oup=512, reduction=coord_reduction) if use_coord_attn else None
        self.agent_attn = AgentAttention2D(
            dim=512,
            num_heads=agent_heads,
            agent_tokens=agent_tokens,
            use_agent_bias=use_agent_bias,
            agent_bias_max_hw=agent_bias_max_hw,
        ) if use_agent_attn else None

        self.fc = nn.Linear(512, num_classes)

    def _apply_attn(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_order == "agent_then_coord":
            if self.agent_attn is not None:
                x = self.agent_attn(x)
            if self.coord_attn is not None:
                x = self.coord_attn(x)
        else:
            if self.coord_attn is not None:
                x = self.coord_attn(x)
            if self.agent_attn is not None:
                x = self.agent_attn(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self._apply_attn(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----------------------------
# Dataset (skip broken images)
# ----------------------------
class TxtImageDataset(Dataset):
    """
    txt 每行格式：<rel_path> <label>
    rel_path 相对 image_root
    """
    def __init__(self, image_root: str, txt_path: str, transform=None, skip_broken: bool = True, bad_log_path: Optional[str] = None):
        self.image_root = image_root
        self.txt_path = txt_path
        self.transform = transform
        self.skip_broken = skip_broken
        self.bad_log_path = bad_log_path
        self.samples: List[Tuple[str, int]] = []
        self.bad_samples: List[str] = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad line in {txt_path}: {line}")
                rel_path = parts[0]
                label = int(parts[1])
                self.samples.append((rel_path, label))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {txt_path}")

        if self.skip_broken:
            kept = []
            for rel_path, label in self.samples:
                p = rel_path if os.path.isabs(rel_path) else os.path.join(self.image_root, rel_path)
                try:
                    if os.path.exists(p) and os.path.getsize(p) == 0:
                        self.bad_samples.append(p)
                        continue
                except Exception:
                    pass
                kept.append((rel_path, label))
            self.samples = kept

        self._flush_bad()

        if len(self.samples) == 0:
            raise ValueError(f"All samples are broken/empty in {txt_path}")

    def _flush_bad(self):
        if self.bad_log_path is None:
            return
        if len(self.bad_samples) == 0:
            return
        try:
            os.makedirs(os.path.dirname(self.bad_log_path), exist_ok=True)
            with open(self.bad_log_path, "a", encoding="utf-8") as f:
                for p in self.bad_samples:
                    f.write(p + "\n")
            self.bad_samples = []
        except Exception:
            self.bad_samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 兼容 txt 里写“绝对路径”和“相对路径”两种格式：
        # - 若 rel_path 是绝对路径：直接使用
        # - 否则：与 image_root 拼接
        #
        # 为了避免“刚好连续遇到坏图”导致训练直接崩，这里把重试次数加大，
        # 并把坏样本记录到 bad_images.txt（由 bad_log_path 控制）。
        for _ in range(200):
            rel_path, label = self.samples[idx]
            img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.image_root, rel_path)
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                if self.transform is not None:
                    im = self.transform(im)
                return im, label
            except Exception:
                if not self.skip_broken:
                    raise
                self.bad_samples.append(img_path)
                self._flush_bad()
                idx = (idx + 1) % len(self.samples)

        raise RuntimeError(
            "Too many broken/unreadable images encountered. "
            "Check bad_images.txt and verify image_root / txt path format."
        )

def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)) -> List[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / targets.size(0)).item())
    return res


@dataclass
class RunConfig:
    image_root: str
    train_txt: str
    val_txt: str
    num_classes: int
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    momentum: float
    weight_decay: float
    print_batch_step: int
    topk: List[int]
    pretrained: bool
    output_dir: str
    seed: int

    # coordinate attention
    use_coord_attn: bool
    coord_reduction: int

    # agent attention
    use_agent_attn: bool
    agent_tokens: int
    agent_heads: int
    use_agent_bias: bool
    agent_bias_max_hw: int

    # stacking order
    attn_order: str

    skip_broken: bool


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="/data")
    parser.add_argument("--train_txt", default="/data/train.txt")
    parser.add_argument("--val_txt", default="/data/val.txt")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--print_batch_step", type=int, default=20)
    parser.add_argument("--topk", default="1,2,4", help="comma separated, e.g. 1,2,4")
    parser.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained weights")
    parser.add_argument("--output_dir", default="./output_torch")
    parser.add_argument("--seed", type=int, default=42)

    # coordinate attention
    parser.add_argument("--use_coord_attn", action="store_true", help="enable Coordinate Attention module after layer4")
    parser.add_argument("--coord_reduction", type=int, default=32, help="reduction ratio r for Coordinate Attention (e.g. 32)")

    # agent attention
    parser.add_argument("--use_agent_attn", action="store_true", help="enable Agent Attention module after layer4")
    parser.add_argument("--agent_tokens", type=int, default=16, help="prefer perfect square, e.g. 16=4x4, 25=5x5")
    parser.add_argument("--agent_heads", type=int, default=8, help="num heads in Agent Attention")
    parser.add_argument("--use_agent_bias", action="store_true", help="enable Agent Bias (B1/B2) for Agent Attention")
    parser.add_argument("--agent_bias_max_hw", type=int, default=32, help="max H/W for relative pos bias table (>= feature map size)")

    # stacking order
    parser.add_argument(
        "--attn_order",
        default="coord_then_agent",
        choices=["coord_then_agent", "agent_then_coord"],
        help="stacking order when both are enabled",
    )

    parser.add_argument("--skip_broken", action="store_true", help="skip broken/empty images (recommended)")

    args = parser.parse_args()
    topk = [int(x) for x in args.topk.split(",") if x.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    cfg = RunConfig(
        image_root=args.image_root,
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        print_batch_step=args.print_batch_step,
        topk=topk,
        pretrained=args.pretrained,
        output_dir=args.output_dir,
        seed=args.seed,
        use_coord_attn=args.use_coord_attn,
        coord_reduction=args.coord_reduction,
        use_agent_attn=args.use_agent_attn,
        agent_tokens=args.agent_tokens,
        agent_heads=args.agent_heads,
        use_agent_bias=args.use_agent_bias,
        agent_bias_max_hw=args.agent_bias_max_hw,
        attn_order=args.attn_order,
        skip_broken=args.skip_broken,
    )

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    bad_log = os.path.join(args.output_dir, "bad_images.txt") if cfg.skip_broken else None
    train_ds = TxtImageDataset(cfg.image_root, cfg.train_txt, transform=train_tf, skip_broken=cfg.skip_broken, bad_log_path=bad_log)
    val_ds = TxtImageDataset(cfg.image_root, cfg.val_txt, transform=val_tf, skip_broken=cfg.skip_broken, bad_log_path=bad_log)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18WithCoordAndAgentAttn(
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        use_coord_attn=cfg.use_coord_attn,
        coord_reduction=cfg.coord_reduction,
        use_agent_attn=cfg.use_agent_attn,
        agent_tokens=cfg.agent_tokens,
        agent_heads=cfg.agent_heads,
        use_agent_bias=cfg.use_agent_bias,
        agent_bias_max_hw=cfg.agent_bias_max_hw,
        attn_order=cfg.attn_order,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    best_top1 = -1.0
    best_path = os.path.join(cfg.output_dir, "best.pt")
    cfg_path = os.path.join(cfg.output_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    print("[Info] device:", device)
    print("[Info] cfg saved:", cfg_path)
    print(
        "[Info] coord_attn:",
        "ON" if cfg.use_coord_attn else "OFF",
        "| coord_reduction:",
        cfg.coord_reduction,
    )
    print(
        "[Info] agent_attn:",
        "ON" if cfg.use_agent_attn else "OFF",
        "| agent_bias:",
        "ON" if (cfg.use_agent_attn and cfg.use_agent_bias) else "OFF",
        "| agent_tokens:",
        cfg.agent_tokens,
        "| heads:",
        cfg.agent_heads,
    )
    print("[Info] attn_order:", cfg.attn_order)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        for it, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            if it % cfg.print_batch_step == 0:
                accs = accuracy_topk(logits.detach(), targets.detach(), topk=tuple(cfg.topk))
                msg = f"[Train][Epoch {epoch}/{cfg.epochs}][Iter {it}/{len(train_loader)}] lr={cfg.lr:.6f}, loss={loss.item():.5f}"
                for k, a in zip(cfg.topk, accs):
                    msg += f", top{k}={a:.5f}"
                print(msg)

        epoch_time = time.time() - t0
        print(f"[Info] epoch_time={epoch_time:.1f}s")

        # eval
        model.eval()
        val_loss = 0.0
        total = 0
        topk_sum = [0.0 for _ in cfg.topk]

        t1 = time.time()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)

                accs = accuracy_topk(logits, targets, topk=tuple(cfg.topk))
                for i, a in enumerate(accs):
                    topk_sum[i] += a * images.size(0)

                total += images.size(0)

        val_loss = val_loss / max(total, 1)
        topk_avg = [x / max(total, 1) for x in topk_sum]

        msg = f"[Eval][Epoch {epoch}] loss={val_loss:.5f}"
        for k, a in zip(cfg.topk, topk_avg):
            msg += f", top{k}={a:.5f}"
        msg += f", time={time.time()-t1:.1f}s"
        print(msg)

        top1 = topk_avg[0] if len(topk_avg) > 0 else -1.0
        if top1 > best_top1:
            best_top1 = top1
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "top1": top1}, best_path)
            print(f"[Info] saved {best_path} (top1={best_top1:.5f})")

    print("[Done] Training finished.")


if __name__ == "__main__":
    main()
