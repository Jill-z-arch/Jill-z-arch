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
# Triplet Attention (Rotate-to-Attend)
# ----------------------------
class ZPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)], dim=1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.compress = ZPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, kernel_size: int = 7, no_spatial: bool = False):
        super().__init__()
        self.no_spatial = no_spatial
        self.cw = SpatialGate(kernel_size=kernel_size)
        self.hc = SpatialGate(kernel_size=kernel_size)
        self.hw = SpatialGate(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x.permute(0, 2, 1, 3).contiguous()
        x1 = self.cw(x1)
        x1 = x1.permute(0, 2, 1, 3).contiguous()

        x2 = x.permute(0, 3, 2, 1).contiguous()
        x2 = self.hc(x2)
        x2 = x2.permute(0, 3, 2, 1).contiguous()

        if self.no_spatial:
            return 0.5 * (x1 + x2)

        x3 = self.hw(x)
        return (x1 + x2 + x3) / 3.0


# ----------------------------
# Agent Attention (保留 only 版本的核心)
# ----------------------------
class AgentAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, agent_tokens: int = 16,
                 use_agent_bias: bool = False, agent_bias_max_hw: int = 32):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        a = int(round(agent_tokens ** 0.5))
        if a * a != agent_tokens:
            a = int((agent_tokens ** 0.5) + 0.9999)
            agent_tokens = a * a
        self.agent_hw = a
        self.agent_tokens = agent_tokens

        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)

        self.agent_q = nn.Linear(dim, dim, bias=False)
        self.agent_k = nn.Linear(dim, dim, bias=False)

        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)

        self.use_agent_bias = use_agent_bias
        self.agent_bias_max_hw = int(agent_bias_max_hw)
        if self.use_agent_bias:
            size = (2 * self.agent_bias_max_hw - 1) * (2 * self.agent_bias_max_hw - 1)
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, size))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        self._bias_index_cache = {}

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _get_agent_coords(self, H: int, W: int, a: int, device) -> torch.Tensor:
        if a == 1:
            ys = torch.tensor([0], device=device, dtype=torch.long)
            xs = torch.tensor([0], device=device, dtype=torch.long)
        else:
            ys = torch.round(torch.linspace(0, H - 1, steps=a, device=device)).long()
            xs = torch.round(torch.linspace(0, W - 1, steps=a, device=device)).long()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)

    def _get_token_coords(self, H: int, W: int, device) -> torch.Tensor:
        ys = torch.arange(H, device=device, dtype=torch.long)
        xs = torch.arange(W, device=device, dtype=torch.long)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)

    def _build_bias_index(self, H: int, W: int, a: int, device):
        if not self.use_agent_bias:
            return None
        key = f"{H}x{W}_a{a}_m{self.agent_bias_max_hw}"
        if key in self._bias_index_cache:
            return self._bias_index_cache[key].to(device)
        token = self._get_token_coords(H, W, device)
        agent = self._get_agent_coords(H, W, a, device)
        rel = token[:, None, :] - agent[None, :, :]
        rel_y = rel[..., 0].clamp(-(self.agent_bias_max_hw - 1), self.agent_bias_max_hw - 1) + (self.agent_bias_max_hw - 1)
        rel_x = rel[..., 1].clamp(-(self.agent_bias_max_hw - 1), self.agent_bias_max_hw - 1) + (self.agent_bias_max_hw - 1)
        size_1d = 2 * self.agent_bias_max_hw - 1
        idx = rel_y * size_1d + rel_x
        self._bias_index_cache[key] = idx.detach().cpu()
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)

        a = self.agent_hw
        pooled = F.adaptive_avg_pool2d(x, (a, a))
        A = pooled.flatten(2).transpose(1, 2)

        Aq = self.agent_q(A)
        Ak = self.agent_k(A)

        qh = self._reshape_heads(q)
        kh = self._reshape_heads(k)
        vh = self._reshape_heads(v)
        Aqh = self._reshape_heads(Aq)
        Akh = self._reshape_heads(Ak)

        bias_index = self._build_bias_index(H, W, a, device=x.device)

        attn1 = (Aqh * self.scale) @ kh.transpose(-2, -1)  # (B,h,A,N)
        if bias_index is not None:
            bias = self.rel_pos_bias[:, bias_index.reshape(-1)].view(self.num_heads, N, a * a)
            attn1 = attn1 + bias.permute(0, 2, 1).unsqueeze(0)

        attn1 = attn1.softmax(dim=-1)
        agent_feat = attn1 @ vh

        attn2 = (qh * self.scale) @ Akh.transpose(-2, -1)  # (B,h,N,A)
        if bias_index is not None:
            bias = self.rel_pos_bias[:, bias_index.reshape(-1)].view(self.num_heads, N, a * a)
            attn2 = attn2 + bias.unsqueeze(0)

        attn2 = attn2.softmax(dim=-1)
        out = attn2 @ agent_feat  # (B,h,N,d)

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.norm(out).transpose(1, 2).contiguous().view(B, C, H, W)

        local = self.dwconv(x)
        return x + out + local


# ----------------------------
# Model: ResNet18 + Triplet + Agent
# ----------------------------
class ResNet18TripletAgent(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        use_triplet_attn: bool,
        triplet_kernel_size: int,
        triplet_no_spatial: bool,
        attn_order: str,
        use_agent_attn: bool,
        agent_tokens: int,
        agent_heads: int,
        use_agent_bias: bool,
        agent_bias_max_hw: int,
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

        self.triplet = TripletAttention(kernel_size=triplet_kernel_size, no_spatial=triplet_no_spatial) if use_triplet_attn else None
        self.agent = AgentAttention2D(dim=512, num_heads=agent_heads, agent_tokens=agent_tokens,
                                      use_agent_bias=use_agent_bias, agent_bias_max_hw=agent_bias_max_hw) if use_agent_attn else None
        self.attn_order = attn_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.triplet is not None and self.agent is not None:
            if self.attn_order == "agent_then_triplet":
                x = self.agent(x)
                x = self.triplet(x)
            else:
                x = self.triplet(x)
                x = self.agent(x)
        elif self.triplet is not None:
            x = self.triplet(x)
        elif self.agent is not None:
            x = self.agent(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


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

    use_triplet_attn: bool
    triplet_kernel_size: int
    triplet_no_spatial: bool
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--print_batch_step", type=int, default=20)
    parser.add_argument("--topk", type=str, default="1,2,4")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--output_dir", type=str, default="/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_broken", action="store_true")

    parser.add_argument("--use_triplet_attn", action="store_true")
    parser.add_argument("--triplet_kernel_size", type=int, default=7)
    parser.add_argument("--triplet_no_spatial", action="store_true")
    parser.add_argument("--attn_order", type=str, default="triplet_then_agent",
                        choices=["triplet_then_agent", "agent_then_triplet"])

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

        use_triplet_attn=args.use_triplet_attn,
        triplet_kernel_size=args.triplet_kernel_size,
        triplet_no_spatial=args.triplet_no_spatial,
        attn_order=args.attn_order,

        use_agent_attn=args.use_agent_attn,
        agent_tokens=args.agent_tokens,
        agent_heads=args.agent_heads,
        use_agent_bias=args.use_agent_bias,
        agent_bias_max_hw=args.agent_bias_max_hw,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    # ---- 强制打印你要求的 meta（也会写入 run_config.json）----
    print(f"[Meta] python_script={os.path.abspath(__file__)}")
    print(f"[Meta] image_root={cfg.image_root}")
    print(f"[Meta] train_txt={cfg.train_txt}")
    print(f"[Meta] val_txt={cfg.val_txt}")
    print(f"[Meta] output_dir={cfg.output_dir}")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18TripletAgent(
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        use_triplet_attn=cfg.use_triplet_attn,
        triplet_kernel_size=cfg.triplet_kernel_size,
        triplet_no_spatial=cfg.triplet_no_spatial,
        attn_order=("agent_then_triplet" if cfg.attn_order == "agent_then_triplet" else "triplet_then_agent"),
        use_agent_attn=cfg.use_agent_attn,
        agent_tokens=cfg.agent_tokens,
        agent_heads=cfg.agent_heads,
        use_agent_bias=cfg.use_agent_bias,
        agent_bias_max_hw=cfg.agent_bias_max_hw,
    ).to(device)

    print(f"[Info] device: {device}")
    print(f"[Info] triplet_attn: {'ON' if cfg.use_triplet_attn else 'OFF'} | kernel={cfg.triplet_kernel_size} | no_spatial={cfg.triplet_no_spatial}")
    print(f"[Info] agent_attn: {'ON' if cfg.use_agent_attn else 'OFF'} | agent_bias={'ON' if cfg.use_agent_bias else 'OFF'} | tokens={cfg.agent_tokens} | heads={cfg.agent_heads}")
    print(f"[Info] attn_order: {cfg.attn_order}")

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

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

        model.eval()
        val_loss = 0.0
        total = 0
        topk_sum = [0.0 for _ in cfg.topk]
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
        print(msg)

        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": asdict(cfg)},
                   os.path.join(cfg.output_dir, "last.pt"))

    print("[Done] training finished.")


if __name__ == "__main__":
    main()
