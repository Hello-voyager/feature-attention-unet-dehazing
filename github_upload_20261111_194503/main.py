import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ========================== 通用工具 ==========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    return torch.device(device_name)


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_checkpoint(path):
    # 新版 PyTorch 默认采用更安全的 weights_only 读取方式。
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def image_to_tensor(image):
    """PIL RGB 图片转为 [3,H,W]、float32、[0,1] 的张量。"""
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1)).copy()
    return torch.from_numpy(array)


def tensor_to_image_array(tensor):
    """[1,3,H,W] 或 [3,H,W] 张量转为 [H,W,3] 的 [0,1] 浮点数组。"""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return array.astype(np.float32)


def array_to_pil(array):
    array_uint8 = np.rint(np.clip(array, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


# ========================== 数据集读取 ==========================

class PairedDehazeDataset(Dataset):
    def __init__(self, data_dir, split, image_size=256, training=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.training = training

        split_dir = self.data_dir / split
        self.hazy_dir = split_dir / "hazy"
        self.clear_dir = split_dir / "clear"

        if not self.hazy_dir.is_dir():
            raise FileNotFoundError(f"找不到有雾图目录：{self.hazy_dir}")
        if not self.clear_dir.is_dir():
            raise FileNotFoundError(f"找不到清晰图目录：{self.clear_dir}")

        self.hazy_paths = sorted(
            p for p in self.hazy_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.clear_paths = sorted(
            p for p in self.clear_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.hazy_paths:
            raise RuntimeError(f"{self.hazy_dir} 中没有图片。")
        if not self.clear_paths:
            raise RuntimeError(f"{self.clear_dir} 中没有图片。")

        self.clear_by_name = {}
        self.clear_by_stem = {}
        for path in self.clear_paths:
            self.clear_by_name.setdefault(path.name, []).append(path)
            self.clear_by_stem.setdefault(path.stem, []).append(path)

        self.pairs = []
        for hazy_path in self.hazy_paths:
            clear_path = self._find_clear_image(hazy_path)
            pair_name = hazy_path.relative_to(self.hazy_dir).as_posix()
            self.pairs.append((hazy_path, clear_path, pair_name))

    def _unique_match(self, candidates, hazy_path):
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(f"有雾图 {hazy_path} 对应多张同名清晰图，请整理文件名。")
        return None

    def _find_clear_image(self, hazy_path):
        # 规则 A：相同相对路径、相同文件名。
        relative_path = hazy_path.relative_to(self.hazy_dir)
        candidate = self.clear_dir / relative_path
        if candidate.is_file():
            return candidate

        # 规则 B：clear 根目录或子目录中有相同文件名。
        match = self._unique_match(self.clear_by_name.get(hazy_path.name, []), hazy_path)
        if match is not None:
            return match

        # 规则 C：RESIDE 常见命名，取有雾图第一个下划线前的场景编号。
        clear_key = hazy_path.stem.split("_")[0]
        match = self._unique_match(self.clear_by_stem.get(clear_key, []), hazy_path)
        if match is not None:
            return match

        raise RuntimeError(
            f"找不到与有雾图对应的清晰图：{hazy_path}\n"
            f"可改成同名配对，例如 hazy/0001.jpg 与 clear/0001.jpg。"
        )

    def __len__(self):
        return len(self.pairs)

    def _load_pair(self, hazy_path, clear_path):
        hazy = Image.open(hazy_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")

        if hazy.size != clear.size:
            # SOTS Indoor：清晰图四周各有10像素白边。
            # 将640×480的清晰图中心裁剪为620×460，与雾图逐像素对齐。
            if (
                    self.split == "test"
                    and hazy.size == (620, 460)
                    and clear.size == (640, 480)
            ):
                clear = clear.crop((10, 10, 630, 470))
            else:
                raise RuntimeError(
                    f"有雾图与清晰图尺寸不一致：{hazy_path.name}，"
                    f"{hazy.size} 与 {clear.size}"
                )

        # 两组模型使用完全相同的预处理方式。
        # 这里统一缩放到固定尺寸，因此自己的指标不要与教材原分辨率结果直接混比。
        resize_method = Image.Resampling.BILINEAR
        target_size = (self.image_size, self.image_size)
        hazy = hazy.resize(target_size, resize_method)
        clear = clear.resize(target_size, resize_method)

        # 只有训练集进行随机水平翻转，并且两张图同步翻转。
        if self.training and random.random() < 0.5:
            hazy = hazy.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            clear = clear.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return image_to_tensor(hazy), image_to_tensor(clear)

    def __getitem__(self, index):
        hazy_path, clear_path, pair_name = self.pairs[index]
        hazy, clear = self._load_pair(hazy_path, clear_path)
        return hazy, clear, pair_name


# ========================== 网络结构 ==========================

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ResidualBlock(nn.Module):
    """教材特征转换模块中的基本残差块：Conv-ReLU-Conv，再与输入相加。"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual


class ChannelAttention(nn.Module):
    """全局平均池化 → 两个 1×1 卷积 → Sigmoid → 通道加权。"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, hidden_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.pool(x)
        weight = self.conv2(self.relu(self.conv1(weight)))
        weight = self.sigmoid(weight)
        return x * weight


class PixelAttention(nn.Module):
    """两个卷积 → Sigmoid → 生成 [N,1,H,W] 空间权重并加权。"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.conv2(self.relu(self.conv1(x)))
        weight = self.sigmoid(weight)
        return x * weight


class FeatureAttention(nn.Module):
    """教材特征注意力：先通道注意力，再像素注意力。"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.pixel_attention = PixelAttention(channels, reduction)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.pixel_attention(x)
        return x


class UpBlock(nn.Module):
    """双线性上采样 + 3×3 卷积；两组模型使用完全相同的上采样方式。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvReLU(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class DehazeUNet(nn.Module):
    """
    类 UNet 去雾网络：3 次下采样、18 个残差块（默认）、3 次上采样。
    use_attention=False：无特征注意力的基线模型。
    use_attention=True ：在三条跳跃连接中加入特征注意力的改进模型。
    """
    def __init__(self, base_channels=32, num_residual=18, reduction=4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        b = base_channels

        self.input_conv = ConvReLU(3, b)
        self.down1 = ConvReLU(b, b * 2, stride=2)
        self.down2 = ConvReLU(b * 2, b * 4, stride=2)
        self.down3 = ConvReLU(b * 4, b * 8, stride=2)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(b * 8) for _ in range(num_residual)]
        )

        # 先创建两组模型共有的层；在相同随机种子下，共有层获得相同初始化。
        self.up2 = UpBlock(b * 8, b * 4)
        self.up1 = UpBlock(b * 4, b * 2)
        self.up0 = UpBlock(b * 2, b)
        self.output_conv = nn.Conv2d(b, 3, kernel_size=3, padding=1)

        # 最后创建注意力层，避免额外模块的随机初始化影响后面的共有层。
        if use_attention:
            self.attention2 = FeatureAttention(b * 4, reduction)
            self.attention1 = FeatureAttention(b * 2, reduction)
            self.attention0 = FeatureAttention(b, reduction)
        else:
            # Identity 不增加参数，保证只去掉特征注意力，不改变其他结构。
            self.attention2 = nn.Identity()
            self.attention1 = nn.Identity()
            self.attention0 = nn.Identity()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("模型输入必须是 [N,3,H,W]。")
        if x.shape[-2] % 8 != 0 or x.shape[-1] % 8 != 0:
            raise ValueError("训练和评价时，图片高宽必须能被 8 整除，例如 256×256。")

        feature0 = self.input_conv(x)       # [N,b,H,W]
        feature1 = self.down1(feature0)     # [N,2b,H/2,W/2]
        feature2 = self.down2(feature1)     # [N,4b,H/4,W/4]
        feature3 = self.down3(feature2)     # [N,8b,H/8,W/8]

        feature3 = self.residual_blocks(feature3)

        skip2 = self.attention2(feature2)
        skip1 = self.attention1(feature1)
        skip0 = self.attention0(feature0)

        decoded2 = self.up2(feature3) + skip2
        decoded1 = self.up1(decoded2) + skip1
        decoded0 = self.up0(decoded1) + skip0
        output = self.output_conv(decoded0)
        return output


# ========================== 训练与验证 ==========================

def run_one_epoch(model, loader, device, optimizer=None, description=""):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=description, leave=False)
    for hazy, clear, _ in progress:
        hazy = hazy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        batch_size = hazy.shape[0]

        with torch.set_grad_enabled(is_training):
            prediction = model(hazy)
            loss = F.smooth_l1_loss(prediction, clear, beta=1.0)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.6f}")

    if total_samples == 0:
        raise RuntimeError("数据加载器没有返回任何样本。")
    return total_loss / total_samples


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, config):
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)


def train_model(args):
    set_seed(args.seed)
    device = get_device(args.device)
    use_attention = args.model == "attention"

    config = {
        "model_type": args.model,
        "use_attention": use_attention,
        "base_channels": args.base_channels,
        "num_residual": args.num_residual,
        "reduction": args.reduction,
        "image_size": args.image_size,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
    }

    if args.image_size <= 0 or args.image_size % 8 != 0:
        raise ValueError("--image-size 必须是大于 0 且能被 8 整除的整数。")

    train_set = PairedDehazeDataset(
        args.data_dir, "train", args.image_size, training=True
    )
    val_set = PairedDehazeDataset(
        args.data_dir, "val", args.image_size, training=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = DehazeUNet(
        base_channels=args.base_channels,
        num_residual=args.num_residual,
        reduction=args.reduction,
        use_attention=use_attention,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = load_checkpoint(resume_path)
        old_config = checkpoint.get("config", {})
        for key in ("use_attention", "base_channels", "num_residual", "reduction"):
            if old_config.get(key) != config[key]:
                raise ValueError(f"恢复训练时模型参数 {key} 与权重文件不一致。")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        run_dir = resume_path.resolve().parent
        best_path = run_dir / "best.pth"
        if best_path.is_file():
            best_checkpoint = load_checkpoint(best_path)
            best_val_loss = float(best_checkpoint.get("val_loss", math.inf))
        else:
            best_val_loss = math.inf
        print(f"从 {resume_path} 恢复训练，下一轮为 Epoch {start_epoch}。")
    else:
        start_epoch = 1
        best_val_loss = math.inf
        run_dir = Path(args.output_dir) / args.run_name
        if (run_dir / "last.pth").exists():
            raise FileExistsError(
                f"{run_dir} 已有训练结果。为避免覆盖，请更换 --run-name，"
                f"或使用 --resume 指定 last.pth。"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    history_path = run_dir / "history.csv"
    if not history_path.exists() or start_epoch == 1:
        with open(history_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "minutes"])

    print("=" * 72)
    print(f"运行设备：{device}")
    if device.type == "cuda":
        print(f"显卡：{torch.cuda.get_device_name(0)}")
    print(f"模型：{args.model}，特征注意力：{use_attention}")
    print(f"训练图片：{len(train_set)}，验证图片：{len(val_set)}")
    print(f"可训练参数量：{count_parameters(model):,}")
    print(f"输出目录：{run_dir}")
    print("=" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = run_one_epoch(
            model, train_loader, device, optimizer,
            description=f"Epoch {epoch}/{args.epochs} 训练"
        )
        with torch.no_grad():
            val_loss = run_one_epoch(
                model, val_loader, device, optimizer=None,
                description=f"Epoch {epoch}/{args.epochs} 验证"
            )

        scheduler.step()
        elapsed_minutes = (time.perf_counter() - epoch_start) / 60.0

        # last.pth 用于中断恢复；best.pth 用于最终测试。
        save_checkpoint(
            run_dir / "last.pth", model, optimizer, scheduler,
            epoch, val_loss, config
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                run_dir / "best.pth", model, optimizer, scheduler,
                epoch, val_loss, config
            )
            best_message = "，已保存新的最佳权重"
        else:
            best_message = ""

        with open(history_path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch, f"{train_loss:.10f}", f"{val_loss:.10f}",
                f"{current_lr:.10f}", f"{elapsed_minutes:.4f}"
            ])

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"lr={current_lr:.8f} | {elapsed_minutes:.2f} 分钟{best_message}"
        )

    print(f"训练结束。最佳验证损失：{best_val_loss:.6f}")
    print(f"最终测试请使用：{run_dir / 'best.pth'}")


# ========================== 测试与指标对比 ==========================

def build_model_from_checkpoint(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    model = DehazeUNet(
        base_channels=int(config["base_channels"]),
        num_residual=int(config["num_residual"]),
        reduction=int(config["reduction"]),
        use_attention=bool(config["use_attention"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint


def warmup(model, sample, device, repeats=3):
    sample = sample.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(repeats):
            model(sample)
    synchronize_if_cuda(device)


def make_comparison_image(hazy, baseline, attention, clear):
    images = [
        array_to_pil(hazy),
        array_to_pil(baseline),
        array_to_pil(attention),
        array_to_pil(clear),
    ]
    width, height = images[0].size
    label_height = 28
    canvas = Image.new("RGB", (width * 4, height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["HAZY", "BASELINE", "ATTENTION", "GROUND TRUTH"]

    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 8, 7), label, fill="black")
    return canvas


def evaluate_model(model, loader, device):
    predictions = []
    metrics = []
    forward_times = []

    with torch.inference_mode():
        for hazy, clear, names in tqdm(loader, desc="测试", leave=False):
            hazy = hazy.to(device)
            clear = clear.to(device)

            synchronize_if_cuda(device)
            start = time.perf_counter()
            prediction = model(hazy)
            synchronize_if_cuda(device)
            forward_times.append((time.perf_counter() - start) * 1000.0)

            hazy_array = tensor_to_image_array(hazy)
            clear_array = tensor_to_image_array(clear)
            prediction_array = tensor_to_image_array(prediction)

            psnr = peak_signal_noise_ratio(
                clear_array, prediction_array, data_range=1.0
            )
            ssim = structural_similarity(
                clear_array, prediction_array,
                data_range=1.0, channel_axis=2
            )

            predictions.append((hazy_array, prediction_array, clear_array))
            metrics.append({
                "name": names[0],
                "psnr": float(psnr),
                "ssim": float(ssim),
            })

    psnr_values = [row["psnr"] for row in metrics]
    ssim_values = [row["ssim"] for row in metrics]
    summary = {
        "count": len(metrics),
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "parameters": count_parameters(model),
        "parameters_million": count_parameters(model) / 1_000_000,
        "inference_ms_mean": float(np.mean(forward_times)),
        "inference_ms_std": float(np.std(forward_times)),
    }
    return metrics, predictions, summary


def evaluate_models(args):
    device = get_device(args.device)
    test_set = PairedDehazeDataset(
        args.data_dir, args.split, args.image_size, training=False
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    baseline_path = Path(args.baseline)
    attention_path = Path(args.attention)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"找不到基线权重：{baseline_path}")
    if not attention_path.is_file():
        raise FileNotFoundError(f"找不到注意力权重：{attention_path}")

    baseline_model, baseline_config, _ = build_model_from_checkpoint(baseline_path, device)
    attention_model, attention_config, _ = build_model_from_checkpoint(attention_path, device)

    if int(baseline_config["image_size"]) != int(attention_config["image_size"]):
        raise ValueError("两组权重训练时使用的 image_size 不一致。")
    if int(baseline_config["image_size"]) != int(args.image_size):
        print(
            f"提示：命令中的 image_size={args.image_size} 与权重训练尺寸 "
            f"{baseline_config['image_size']} 不同。当前评价仍使用命令中的尺寸。"
        )

    first_hazy, _, _ = test_set[0]
    warmup(baseline_model, first_hazy, device)
    warmup(attention_model, first_hazy, device)

    print("正在测试无注意力基线模型……")
    baseline_metrics, baseline_predictions, baseline_summary = evaluate_model(
        baseline_model, test_loader, device
    )
    print("正在测试有特征注意力的改进模型……")
    attention_metrics, attention_predictions, attention_summary = evaluate_model(
        attention_model, test_loader, device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "comparisons"
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (base_row, att_row) in enumerate(zip(baseline_metrics, attention_metrics)):
        rows.append({
            "name": base_row["name"],
            "baseline_psnr": base_row["psnr"],
            "baseline_ssim": base_row["ssim"],
            "attention_psnr": att_row["psnr"],
            "attention_ssim": att_row["ssim"],
            "delta_psnr": att_row["psnr"] - base_row["psnr"],
            "delta_ssim": att_row["ssim"] - base_row["ssim"],
        })

        if args.save_images:
            hazy_array = baseline_predictions[index][0]
            baseline_array = baseline_predictions[index][1]
            attention_array = attention_predictions[index][1]
            clear_array = baseline_predictions[index][2]
            comparison = make_comparison_image(
                hazy_array, baseline_array, attention_array, clear_array
            )
            safe_stem = Path(base_row["name"]).stem.replace("/", "_").replace("\\", "_")
            comparison.save(images_dir / f"{index:04d}_{safe_stem}.png")

    with open(output_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "test_split": args.split,
        "image_size": args.image_size,
        "baseline_checkpoint": str(baseline_path),
        "attention_checkpoint": str(attention_path),
        "baseline": baseline_summary,
        "attention": attention_summary,
        "improvement": {
            "delta_psnr_mean": attention_summary["psnr_mean"] - baseline_summary["psnr_mean"],
            "delta_ssim_mean": attention_summary["ssim_mean"] - baseline_summary["ssim_mean"],
            "delta_parameters": attention_summary["parameters"] - baseline_summary["parameters"],
            "delta_inference_ms": attention_summary["inference_ms_mean"] - baseline_summary["inference_ms_mean"],
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"测试图片数：{baseline_summary['count']}")
    print(f"{'模型':<14}{'PSNR↑':>12}{'SSIM↑':>12}{'参数量/M':>14}{'时间/ms':>14}")
    print(
        f"{'Baseline':<14}{baseline_summary['psnr_mean']:>12.4f}"
        f"{baseline_summary['ssim_mean']:>12.6f}"
        f"{baseline_summary['parameters_million']:>14.4f}"
        f"{baseline_summary['inference_ms_mean']:>14.3f}"
    )
    print(
        f"{'Attention':<14}{attention_summary['psnr_mean']:>12.4f}"
        f"{attention_summary['ssim_mean']:>12.6f}"
        f"{attention_summary['parameters_million']:>14.4f}"
        f"{attention_summary['inference_ms_mean']:>14.3f}"
    )
    print("=" * 78)
    print(f"逐图指标：{output_dir / 'per_image_metrics.csv'}")
    print(f"汇总指标：{output_dir / 'summary.json'}")
    if args.save_images:
        print(f"对比图片：{images_dir}")


# ========================== 真实雾图推理 ==========================

def pad_to_multiple(tensor, multiple=8):
    """对单张 [1,3,H,W] 图片右侧、下侧填充，推理后再裁掉。"""
    height, width = tensor.shape[-2:]
    new_height = max(multiple, math.ceil(height / multiple) * multiple)
    new_width = max(multiple, math.ceil(width / multiple) * multiple)
    pad_bottom = new_height - height
    pad_right = new_width - width

    if pad_bottom == 0 and pad_right == 0:
        return tensor, height, width

    mode = "reflect"
    if pad_bottom >= height or pad_right >= width:
        mode = "replicate"
    tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode=mode)
    return tensor, height, width


def collect_inference_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise RuntimeError(f"{input_path} 中没有可处理的图片。")
        return paths
    raise FileNotFoundError(f"找不到输入图片或目录：{input_path}")


def infer_images(args):
    device = get_device(args.device)
    model, _, _ = build_model_from_checkpoint(args.checkpoint, device)
    image_paths = collect_inference_images(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for path in tqdm(image_paths, desc="真实雾图去雾"):
            image = Image.open(path).convert("RGB")
            tensor = image_to_tensor(image).unsqueeze(0).to(device)
            padded, original_height, original_width = pad_to_multiple(tensor, 8)
            prediction = model(padded)
            prediction = prediction[:, :, :original_height, :original_width]
            result = array_to_pil(tensor_to_image_array(prediction))
            output_path = output_dir / f"{path.stem}_dehazed.png"
            result.save(output_path)
            print(f"已保存：{output_path}")


# ========================== 检查命令 ==========================

def check_data(args):
    for split in ("train", "val", "test"):
        dataset = PairedDehazeDataset(
            args.data_dir, split, args.image_size, training=False
        )
        print(f"{split}：{len(dataset)} 组")
        for hazy_path, clear_path, _ in dataset.pairs[:3]:
            print(f"  {hazy_path.name}  <->  {clear_path.name}")
        hazy, clear, name = dataset[0]
        print(f"  首组张量：hazy={tuple(hazy.shape)}，clear={tuple(clear.shape)}，名称={name}")
        if hazy.shape != clear.shape:
            raise RuntimeError(f"{split} 首组张量形状不一致。")
    print("训练、验证、测试图片均能读取，配对和尺寸检查通过。")


def smoke_test(args):
    device = get_device(args.device)
    print(f"测试设备：{device}")
    x = torch.rand(2, 3, 64, 64, device=device)
    target = torch.rand(2, 3, 64, 64, device=device)

    for use_attention in (False, True):
        model = DehazeUNet(
            base_channels=8, num_residual=2, reduction=4,
            use_attention=use_attention
        ).to(device)
        prediction = model(x)
        assert prediction.shape == x.shape, (prediction.shape, x.shape)
        loss = F.smooth_l1_loss(prediction, target)
        loss.backward()
        print(
            f"use_attention={use_attention}，输出={tuple(prediction.shape)}，"
            f"loss={loss.item():.6f}，参数量={count_parameters(model):,}，反向传播通过。"
        )
    print("两组模型的前向传播、损失计算和反向传播均通过。")


# ========================== 命令行入口 ==========================

def build_parser():
    parser = argparse.ArgumentParser(
        description="基于特征注意力类 UNet 的图像去雾：训练、对比评价与真实图片推理。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_data_arguments(subparser):
        subparser.add_argument(
            "--data-dir", type=Path, default=ROOT_DIR / "data",
            help="数据根目录，默认是代码旁边的 data。"
        )
        subparser.add_argument("--image-size", type=int, default=256)
        subparser.add_argument(
            "--device", choices=["auto", "cuda", "cpu"], default="auto"
        )

    check_parser = subparsers.add_parser("check-data", help="检查三个数据集分区的配对与读取")
    add_common_data_arguments(check_parser)

    smoke_parser = subparsers.add_parser("smoke", help="不读数据，快速检查两组模型")
    smoke_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )

    train_parser = subparsers.add_parser("train", help="训练基线或注意力模型")
    add_common_data_arguments(train_parser)
    train_parser.add_argument("--model", choices=["baseline", "attention"], required=True)
    train_parser.add_argument("--run-name", default=None, help="默认与 --model 相同")
    train_parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "runs")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--base-channels", type=int, default=32)
    train_parser.add_argument("--num-residual", type=int, default=18)
    train_parser.add_argument("--reduction", type=int, default=4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--resume", default=None, help="可填写 runs/模型名/last.pth")

    eval_parser = subparsers.add_parser("evaluate", help="对比两组模型的 PSNR、SSIM、参数量和速度")
    add_common_data_arguments(eval_parser)
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument(
        "--baseline", default=str(ROOT_DIR / "runs" / "baseline" / "best.pth")
    )
    eval_parser.add_argument(
        "--attention", default=str(ROOT_DIR / "runs" / "attention" / "best.pth")
    )
    eval_parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results")
    eval_parser.add_argument(
        "--save-images", action=argparse.BooleanOptionalAction, default=True,
        help="默认保存四列对比图；可用 --no-save-images 关闭。"
    )

    infer_parser = subparsers.add_parser("infer", help="处理无清晰真值的真实雾图")
    infer_parser.add_argument("--input", required=True, help="图片文件或图片目录")
    infer_parser.add_argument(
        "--checkpoint", default=str(ROOT_DIR / "runs" / "attention" / "best.pth")
    )
    infer_parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "inference")
    infer_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train" and args.run_name is None:
        args.run_name = args.model

    if args.command == "check-data":
        check_data(args)
    elif args.command == "smoke":
        smoke_test(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    elif args.command == "infer":
        infer_images(args)
    else:
        parser.error("未知命令。")


if __name__ == "__main__":
    main()