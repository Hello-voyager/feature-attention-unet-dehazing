import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def read_history(model_name):
    path = ROOT / "runs" / model_name / "history.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"{model_name}：读取到{len(rows)}轮训练记录，来源：{path}")

    return {
        "epoch": [int(float(row["epoch"])) for row in rows],
        "train_loss": [float(row["train_loss"]) for row in rows],
        "val_loss": [float(row["val_loss"]) for row in rows],
    }


def print_metrics():
    path = ROOT / "results" / "summary.json"
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\n========== 500张测试图的指标对比 ==========")
    print(f"结果来源：{path}")

    for key, name in [
        ("baseline", "无注意力基线"),
        ("attention", "特征注意力"),
    ]:
        result = summary[key]
        print(
            f"{name}：\n"
            f"  PSNR = {result['psnr_mean']:.6f} ± "
            f"{result['psnr_std']:.6f} dB\n"
            f"  SSIM = {result['ssim_mean']:.6f} ± "
            f"{result['ssim_std']:.6f}\n"
            f"  参数量 = {result['parameters']:,}\n"
            f"  前向时间 = {result['inference_ms_mean']:.6f} ms\n"
        )

    improvement = summary["improvement"]
    print("注意力相对基线的变化：")
    print(f"  PSNR变化：{improvement['delta_psnr_mean']:.6f} dB")
    print(f"  SSIM变化：{improvement['delta_ssim_mean']:.6f}")
    print(f"  参数变化：{improvement['delta_parameters']:+,}")
    print(f"  时间变化：{improvement['delta_inference_ms']:+.6f} ms")


def show_loss_curves():
    baseline = read_history("baseline")
    attention = read_history("attention")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for ax, key, title in [
        (axes[0], "train_loss", "训练损失"),
        (axes[1], "val_loss", "验证损失"),
    ]:
        ax.plot(
            baseline["epoch"],
            baseline[key],
            color="#355C67",
            linewidth=1.8,
            label="无注意力基线",
        )
        ax.plot(
            attention["epoch"],
            attention[key],
            color="#A67536",
            linewidth=1.8,
            label="特征注意力",
        )
        ax.set_title(title)
        ax.set_xlabel("训练轮数（Epoch）")
        ax.set_ylabel("Smooth L1 损失")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    axes[0].set_yscale("log")

    for data, name, color in [
        (baseline, "基线", "#355C67"),
        (attention, "注意力", "#A67536"),
    ]:
        best_index = min(
            range(len(data["val_loss"])),
            key=lambda i: data["val_loss"][i],
        )
        best_epoch = data["epoch"][best_index]
        best_loss = data["val_loss"][best_index]

        axes[1].scatter(
            best_epoch, best_loss,
            color=color, s=35, zorder=3,
        )
        print(
            f"{name}最佳验证结果：第{best_epoch}轮，"
            f"损失={best_loss:.10f}"
        )

    fig.suptitle("两组模型均完成100轮训练")
    fig.tight_layout()

    output = ROOT / "results" / "loss_curves.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Loss曲线已保存：{output}")

    plt.show()


def show_comparison_images():
    images = [
        (
            "0380_1438_1.png",
            "改善样本：PSNR提高3.890955 dB，SSIM提高0.002594",
        ),
        (
            "0323_1432_3.png",
            "退化样本：PSNR下降6.343323 dB，SSIM下降0.024103",
        ),
        (
            "0321_1432_10.png",
            "退化样本：PSNR下降6.298613 dB，SSIM下降0.055985",
        ),
    ]

    for filename, title in images:
        path = ROOT / "results" / "comparisons" / filename
        print(f"展示对比图：{path}")

        with Image.open(path) as image:
            plt.figure(figsize=(14, 4.2))
            plt.imshow(image)
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    print_metrics()
    show_loss_curves()
    show_comparison_images()
