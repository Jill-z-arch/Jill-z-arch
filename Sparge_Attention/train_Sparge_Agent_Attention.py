#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models
from PIL import Image


# ----------------------------
# utils
# ----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# ----------------------------
# Dataset
# ----------------------------
class TxtImageDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        txt_path: str,
        transform=None,
        skip_broken: bool = False,
        bad_log_path: Optional[str] = None,
    ):
        self.image_root = image_root
        self.txt_path = txt_path
        self.transform = transform
        self.skip_broken = skip_broken
        self.bad_log_path = bad_log_path

        self.samples: List[Tuple[str, int]] = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    continue
                self.samples.append((parts[0], int(parts[1])))

        if self.bad_log_path:
            os.makedirs(os.path.dirname(self.bad_log_path), exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _log_bad(self, path: str):
        if not self.bad_log_path:
            return
        try:
            with open(self.bad_log_path, "a", encoding="utf-8") as f:
                f.write(path + "\n")
        except Exception:
            pass

    def __getitem__(self, idx):
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
                self._log_bad(img_path)
                idx = (idx + 1) % len(self.samples)

        raise RuntimeError(
            "Too many broken/unreadable images encountered. "
            "Check bad_images.txt and verify image_root / txt path format."
        )


def quick_path_check(image_root: str, txt_path: str, n: int = 5):
    # 避免“路径错导致看似卡住/全坏图”这种坑
    ok = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    for ln in lines[: max(n, 1)]:
        p = ln.split()[0]
        img = p if os.path.isabs(p) else os.path.join(image_root, p)
        if os.path.isfile(img):
            ok += 1
    print(f"[Info] path_check: {ok}/{min(len(lines), n)} exist for first {min(len(lines), n)} samples")


# ----------------------------
# Agent Attention (保留)
# ----------------------------
class AgentAttention2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        agent_tokens: int = 16,
        use_agent_bias: bool = True,
        agent_bias_max_hw: int = 32,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.agent_tokens = agent_tokens
        self.use_agent_bias = use_agent_bias
        self.agent_bias_max_hw = agent_bias_max_hw

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        if use_agent_bias:
            self.bias1 = nn.Parameter(torch.zeros(num_heads, agent_tokens, agent_bias_max_hw * agent_bias_max_hw))
            self.bias2 = nn.Parameter(torch.zeros(num_heads, agent_bias_max_hw * agent_bias_max_hw, agent_tokens))
        else:
            self.bias1 = None
            self.bias2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        n = h * w
        q = q.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)  # [B, heads, N, d]
        k = k.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        v = v.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)

        a = self.agent_tokens
        ah = max(1, int(round(a ** 0.5)))
        aw = max(1, a // ah)
        pooled = F.adaptive_avg_pool2d(x, (ah, aw))
        A = pooled.reshape(b, self.num_heads, self.head_dim, ah * aw).transpose(2, 3)  # [B, heads, A, d]
        A_len = A.shape[2]

        score1 = torch.matmul(A, k.transpose(-2, -1))  # [B, heads, A, N]
        score2 = torch.matmul(q, A.transpose(-2, -1))  # [B, heads, N, A]

        if self.use_agent_bias:
            hw_cap = self.agent_bias_max_hw
            hs = torch.linspace(0, hw_cap - 1, steps=h, device=x.device).long()
            ws = torch.linspace(0, hw_cap - 1, steps=w, device=x.device).long()
            grid = (hs[:, None] * hw_cap + ws[None, :]).reshape(-1)

            b1 = self.bias1[:, :A_len, :].index_select(-1, grid)  # [heads, A, N]
            b2 = self.bias2[:, :, :A_len].index_select(1, grid)   # [heads, N, A]
            score1 = score1 + b1.unsqueeze(0)
            score2 = score2 + b2.unsqueeze(0)

        attn1 = F.softmax(score1, dim=-1)
        attn2 = F.softmax(score2, dim=-1)

        AV = torch.matmul(attn1, v)
        out = torch.matmul(attn2, AV)
        out = out.transpose(2, 3).reshape(b, c, h, w)

        out = out + self.dwc(x)
        out = self.proj(out)
        return out


# ----------------------------
# Sparge Attention (Top-K 稀疏注意力)
# ----------------------------
class SpargeAttention2D(nn.Module):
    """
    2D feature map attention with Top-K sparsification over spatial tokens.
    - Input: [B, C, H, W]
    - Output: [B, C, H, W]
    """
    def __init__(self, dim: int, num_heads: int = 8, topk: int = 64):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.topk = topk

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)  # [B, heads, N, d]
        k = k.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        v = v.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, N, N]

        # Top-K sparsify on last dim
        k_top = min(self.topk, n)
        vals, idx = torch.topk(attn, k=k_top, dim=-1, largest=True, sorted=False)
        sparse = torch.full_like(attn, float("-inf"))
        sparse.scatter_(-1, idx, vals)

        attn = F.softmax(sparse, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, N, d]
        out = out.transpose(2, 3).reshape(b, c, h, w)
        out = self.proj(out)
        return out


# ----------------------------
# Model: ResNet18 + (Sparge) + (Agent)
# ----------------------------
class ResNet18WithAttn(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        use_sparge_attn: bool,
        sparge_topk: int,
        use_agent_attn: bool,
        agent_tokens: int,
        agent_heads: int,
        use_agent_bias: bool,
        agent_bias_max_hw: int,
        attn_order: str = "sparge_then_agent",
    ):
        super().__init__()
        if pretrained:
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = models.resnet18(pretrained=True)
        else:
            try:
                backbone = models.resnet18(weights=None)
            except Exception:
                backbone = models.resnet18(pretrained=False)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

        self.use_sparge_attn = use_sparge_attn
        self.use_agent_attn = use_agent_attn
        self.attn_order = attn_order

        self.sparge = SpargeAttention2D(dim=512, num_heads=8, topk=sparge_topk) if use_sparge_attn else None
        self.agent = AgentAttention2D(
            dim=512,
            num_heads=agent_heads,
            agent_tokens=agent_tokens,
            use_agent_bias=use_agent_bias,
            agent_bias_max_hw=agent_bias_max_hw,
        ) if use_agent_attn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_sparge_attn and self.use_agent_attn:
            if self.attn_order == "sparge_then_agent":
                x = self.sparge(x)
                x = self.agent(x)
            else:
                x = self.agent(x)
                x = self.sparge(x)
        elif self.use_sparge_attn:
            x = self.sparge(x)
        elif self.use_agent_attn:
            x = self.agent(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----------------------------
# Config
# ----------------------------
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
    skip_broken: bool

    use_sparge_attn: bool
    sparge_topk: int
    attn_order: str

    use_agent_attn: bool
    agent_tokens: int
    agent_heads: int
    use_agent_bias: bool
    agent_bias_max_hw: int


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--train_txt", type=str, required=True)
    parser.add_argument("--val_txt", type=str, required=True)

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--print_batch_step", type=int, default=20)
    parser.add_argument("--topk", type=str, default="1,2,4")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--output_dir", type=str, default="/mnt/synology/wen.zhou/2026.05/Sparge_Attention/output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_broken", action="store_true")

    # Sparge
    parser.add_argument("--use_sparge_attn", action="store_true")
    parser.add_argument("--sparge_topk", type=int, default=64)
    parser.add_argument("--attn_order", type=str, default="sparge_then_agent",
                        choices=["sparge_then_agent", "agent_then_sparge"])

    # Agent
    parser.add_argument("--use_agent_attn", action="store_true")
    parser.add_argument("--agent_tokens", type=int, default=16)
    parser.add_argument("--agent_heads", type=int, default=8)
    parser.add_argument("--use_agent_bias", action="store_true")
    parser.add_argument("--agent_bias_max_hw", type=int, default=32)

    args = parser.parse_args()
    topk = [int(x) for x in args.topk.split(",") if x.strip()]

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
        skip_broken=args.skip_broken,

        use_sparge_attn=args.use_sparge_attn,
        sparge_topk=args.sparge_topk,
        attn_order=args.attn_order,

        use_agent_attn=args.use_agent_attn,
        agent_tokens=args.agent_tokens,
        agent_heads=args.agent_heads,
        use_agent_bias=args.use_agent_bias,
        agent_bias_max_hw=args.agent_bias_max_hw,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")
    print(f"[Info] sparge_attn: {'ON' if cfg.use_sparge_attn else 'OFF'} | sparge_topk={cfg.sparge_topk}")
    print(f"[Info] agent_attn: {'ON' if cfg.use_agent_attn else 'OFF'} | agent_tokens={cfg.agent_tokens} | heads={cfg.agent_heads}")
    print(f"[Info] attn_order: {cfg.attn_order}")

    cfg_path = os.path.join(cfg.output_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    print(f"[Info] cfg saved: {cfg_path}")

    quick_path_check(cfg.image_root, cfg.train_txt, n=5)

    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    bad_log = os.path.join(cfg.output_dir, "bad_images.txt")
    train_ds = TxtImageDataset(cfg.image_root, cfg.train_txt, transform=train_tf,
                               skip_broken=cfg.skip_broken, bad_log_path=bad_log)
    val_ds = TxtImageDataset(cfg.image_root, cfg.val_txt, transform=val_tf,
                             skip_broken=cfg.skip_broken, bad_log_path=bad_log)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = ResNet18WithAttn(
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        use_sparge_attn=cfg.use_sparge_attn,
        sparge_topk=cfg.sparge_topk,
        use_agent_attn=cfg.use_agent_attn,
        agent_tokens=cfg.agent_tokens,
        agent_heads=cfg.agent_heads,
        use_agent_bias=cfg.use_agent_bias,
        agent_bias_max_hw=cfg.agent_bias_max_hw,
        attn_order=cfg.attn_order,
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_top1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()

        for it, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            if cfg.print_batch_step > 0 and (it % cfg.print_batch_step == 0):
                lr_now = optimizer.param_groups[0]["lr"]
                accs = accuracy_topk(logits.detach(), targets.detach(), topk=tuple(cfg.topk))
                msg = f"[Train][Epoch {epoch}/{cfg.epochs}][Iter {it}/{len(train_loader)}] lr={lr_now:.6f}, loss={loss.item():.5f}"
                for k, a in zip(cfg.topk, accs):
                    msg += f", top{k}={a:.5f}"
                print(msg)

        print(f"[Info] epoch_time={time.time()-t0:.1f}s")

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

        val_loss = val_loss / max(1, total)
        val_accs = [x / max(1, total) for x in topk_sum]
        msg = f"[Val][Epoch {epoch}/{cfg.epochs}] loss={val_loss:.5f}"
        for k, a in zip(cfg.topk, val_accs):
            msg += f", top{k}={a:.5f}"
        msg += f", time={time.time()-t1:.1f}s"
        print(msg)

        top1 = val_accs[0] if len(val_accs) > 0 else 0.0
        if top1 > best_top1:
            best_top1 = top1
            ckpt_path = os.path.join(cfg.output_dir, "best.pt")
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)
            print(f"[Info] saved best: {ckpt_path} (top1={best_top1:.5f})")

    last_path = os.path.join(cfg.output_dir, "last.pt")
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, last_path)
    print(f"[Info] saved last: {last_path}")


if __name__ == "__main__":
    main()
