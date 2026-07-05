import argparse
import copy
import glob
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm
from PIL import Image
import csv
import json
import os
import yaml
from datetime import datetime
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def resolve_path(path):
    if not path or os.path.isabs(path):
        return path
    candidates = [
        os.path.normpath(os.path.join(SCRIPT_DIR, path)),
        os.path.normpath(os.path.join(PROJECT_ROOT, path)),
        os.path.normpath(os.path.join(os.getcwd(), path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def cifar100_exists(root):
    return (
        os.path.isfile(os.path.join(root, "cifar-100-python", "train"))
        and os.path.isfile(os.path.join(root, "cifar-100-python", "test"))
    )


parser = argparse.ArgumentParser()
parser.add_argument("-new", action="store_true")
parser.add_argument("-resume", metavar="PTH", default=None)
parser.add_argument("-test", metavar="PTH", default=None)
args = parser.parse_args()

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config_v2.yaml")
with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SEED = cfg.get("seed", 42)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

EPOCHS = cfg["epochs"]
BATCH_SIZE = cfg["batch_size"]
IMG_SIZE = cfg["img_size"]
NUM_CLASSES = cfg["num_classes"]
CIFAR100_ROOT = resolve_path(cfg.get("cifar100_root", "../dlp/data"))
CIMAGES_DIR = resolve_path(cfg.get("cimages_dir", "../announcement/CImages_dataset/CImages"))
LOG_DIR = resolve_path(cfg.get("log_dir", "log"))
MODEL_LOG_DIR = os.path.join(LOG_DIR, "model")
TEST_LOG_DIR = os.path.join(LOG_DIR, "test")
NUM_WORKERS = int(cfg.get("num_workers", 2))
MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
MODEL_TAG = cfg.get("model_tag", "convnextv2_dsaf_bottleneck")
FUSION_OUT_INDICES = tuple(cfg.get("fusion_out_indices", [2, 3]))
FUSION_DIM = int(cfg.get("fusion_dim", 384))
BOTTLENECK_DIM = int(cfg.get("bottleneck_dim", 256))
DROPOUT = float(cfg.get("dropout", 0.2))
AUX_LOSS_WEIGHT = float(cfg.get("aux_loss_weight", 0.15))
MAX_PARAMS = int(cfg.get("max_params", 30_000_000))

# 🔥 2번에서 가져온 학습전략 하이퍼파라미터
EMA_DECAY = 0.9999
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 0.8
MIXUP_PROB = 0.4   # 40% mixup, 60% cutmix
WARMUP_EPOCHS = 5
DROP_PATH_RATE = 0.1  # 2번과 동일하게 0.1로 수정

if len(FUSION_OUT_INDICES) != 2:
    raise ValueError(f"fusion_out_indices must contain exactly 2 stages: {FUSION_OUT_INDICES}")

device = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = device == "cuda"
print(f"Using device: {device}")
print(f"CIFAR-100 root: {CIFAR100_ROOT}")

# ─────────────────────────────────────────
# 데이터 전처리 (2번 augmentation 적용)
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),  # 🔥 2번과 동일
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),  # 🔥 2번에서 추가
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CImagesDataset(Dataset):
    def __init__(self, root, transform=None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"CImages directory not found: {root}")
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg files found in CImages directory: {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        number = os.path.splitext(os.path.basename(self.paths[idx]))[0]
        return img, number


cifar100_download = not cifar100_exists(CIFAR100_ROOT)
if cifar100_download:
    print("CIFAR-100 files not found locally. torchvision will try to download them.")

# ✅ Test Leakage 방지: train=True만 학습에 사용, train=False는 evaluate()에서만 사용
train_dataset = datasets.CIFAR100(
    root=CIFAR100_ROOT, train=True, download=cifar100_download, transform=train_transform,
)
cifar_test_dataset = datasets.CIFAR100(
    root=CIFAR100_ROOT, train=False, download=cifar100_download, transform=test_transform,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)
# ✅ shuffle=False 고정: 평가 전용, 학습에 절대 사용 안 함
cifar_test_loader = DataLoader(
    cifar_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)


# ─────────────────────────────────────────
# Mixup / CutMix (2번에서 이식)
# ─────────────────────────────────────────
def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix(x, y, alpha=CUTMIX_ALPHA):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = x.shape
    idx = torch.randperm(B).to(x.device)
    cut_h = int(H * (1 - lam) ** 0.5)
    cut_w = int(W * (1 - lam) ** 0.5)
    cx = torch.randint(H, (1,)).item()
    cy = torch.randint(W, (1,)).item()
    x1, x2 = max(cx - cut_h // 2, 0), min(cx + cut_h // 2, H)
    y1, y2 = max(cy - cut_w // 2, 0), min(cy + cut_w // 2, W)
    x_cut = x.clone()
    x_cut[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    return x_cut, y, y[idx], lam


def mixup_cutmix_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────
# 학습 / 평가 함수
# ─────────────────────────────────────────
def compute_loss_and_logits(model, X, y, criterion, mixup_args=None):
    outputs = model(X, return_aux=getattr(model, "aux_loss_weight", 0.0) > 0)

    if mixup_args:
        y_a, y_b, lam = mixup_args
        def calculate_loss(logits):
            return mixup_cutmix_loss(criterion, logits, y_a, y_b, lam)
    else:
        def calculate_loss(logits):
            return criterion(logits, y)

    if getattr(model, "aux_loss_weight", 0.0) > 0 and isinstance(outputs, dict):
        main_loss = calculate_loss(outputs["logits"])
        stage3_loss = calculate_loss(outputs["stage3_logits"])
        stage4_loss = calculate_loss(outputs["stage4_logits"])
        loss = main_loss + model.aux_loss_weight * (stage3_loss + stage4_loss)
        return loss, outputs["logits"]

    logits = outputs if not isinstance(outputs, dict) else outputs["logits"]
    return calculate_loss(logits), logits


def train_one_epoch(model, loader, optimizer, criterion, ema_model=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # 🔥 Mixup 40% / CutMix 60% (2번과 동일)
        if torch.rand(1).item() < MIXUP_PROB:
            X, y_a, y_b, lam = mixup(X, y)
        else:
            X, y_a, y_b, lam = cutmix(X, y)

        loss, pred = compute_loss_and_logits(
            model, X, y, criterion, mixup_args=(y_a, y_b, lam)
        )
        loss.backward()
        optimizer.step()

        # 🔥 EMA 업데이트 (2번에서 이식)
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)

        total_loss += loss.item()
        correct += (pred.argmax(1) == y_a).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    """✅ cifar_test_loader는 여기서만 사용. 학습 루프에 절대 포함 안 됨."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


def report_parameter_count(model):
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1_000_000:.2f}M / {MAX_PARAMS / 1_000_000:.1f}M")
    if param_count > MAX_PARAMS:
        raise RuntimeError(f"Parameter limit exceeded: {param_count:,} > {MAX_PARAMS:,}")
    return param_count


# ─────────────────────────────────────────
# 모델 정의 (1번 구조 유지)
# ─────────────────────────────────────────
class AdaptiveStage34FusionConvNeXtV2(nn.Module):
    def __init__(self, pretrained, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=FUSION_OUT_INDICES,
            drop_path_rate=DROP_PATH_RATE,  # 🔥 0.2 → 0.1로 수정
        )
        stage_channels = self.backbone.feature_info.channels()
        if len(stage_channels) != 2:
            raise RuntimeError(f"Expected 2 fusion stages, got {stage_channels}")

        self.stage3_channels, self.stage4_channels = stage_channels
        self.fusion_dim = FUSION_DIM
        self.aux_loss_weight = AUX_LOSS_WEIGHT

        self.stage3_projection = nn.Sequential(
            nn.LayerNorm(self.stage3_channels * 2),
            nn.Linear(self.stage3_channels * 2, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.stage4_projection = nn.Sequential(
            nn.LayerNorm(self.stage4_channels * 2),
            nn.Linear(self.stage4_channels * 2, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        gate_hidden_dim = max(self.fusion_dim // 2, 64)
        self.stage_gate = nn.Sequential(
            nn.LayerNorm(self.fusion_dim * 2),
            nn.Linear(self.fusion_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 2),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, BOTTLENECK_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(BOTTLENECK_DIM, num_classes),
        )
        self.stage3_aux_classifier = nn.Linear(self.fusion_dim, num_classes)
        self.stage4_aux_classifier = nn.Linear(self.fusion_dim, num_classes)

    def _stat_pool_feature(self, feature, channels):
        if feature.ndim != 4:
            raise RuntimeError(f"Expected 4D feature map, got shape {feature.shape}")
        if feature.shape[1] == channels:
            avg_feature = feature.mean(dim=(2, 3))
            max_feature = feature.amax(dim=(2, 3))
            return torch.cat([avg_feature, max_feature], dim=1)
        if feature.shape[-1] == channels:
            avg_feature = feature.mean(dim=(1, 2))
            max_feature = feature.amax(dim=(1, 2))
            return torch.cat([avg_feature, max_feature], dim=1)
        raise RuntimeError(
            f"Cannot infer feature format for shape {feature.shape}, channels={channels}"
        )

    def forward(self, x, return_aux=False):
        stage3_feature, stage4_feature = self.backbone(x)
        stage3_feature = self.stage3_projection(
            self._stat_pool_feature(stage3_feature, self.stage3_channels)
        )
        stage4_feature = self.stage4_projection(
            self._stat_pool_feature(stage4_feature, self.stage4_channels)
        )
        gate_logits = self.stage_gate(torch.cat([stage3_feature, stage4_feature], dim=1))
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused_feature = (
            gate_weights[:, 0:1] * stage3_feature
            + gate_weights[:, 1:2] * stage4_feature
        )
        logits = self.classifier(fused_feature)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "stage3_logits": self.stage3_aux_classifier(stage3_feature),
            "stage4_logits": self.stage4_aux_classifier(stage4_feature),
            "gate_weights": gate_weights,
        }


def find_latest_checkpoint(exp_name):
    matches = sorted(
        glob.glob(os.path.join(MODEL_LOG_DIR, f"*_{MODEL_TAG}_{exp_name}_checkpoint.pth"))
    )
    return matches[-1] if matches else None


# ─────────────────────────────────────────
# 테스트 모드
# ─────────────────────────────────────────
RUN_DT = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(MODEL_LOG_DIR, exist_ok=True)
os.makedirs(TEST_LOG_DIR, exist_ok=True)

if args.test:
    model = AdaptiveStage34FusionConvNeXtV2(pretrained=False, num_classes=NUM_CLASSES).to(device)
    report_parameter_count(model)
    state = torch.load(args.test, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    cimages_loader = DataLoader(
        CImagesDataset(CIMAGES_DIR, transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    import time
    predictions, confidences = [], []
    t0 = time.time()
    with torch.no_grad():
        for X, numbers in tqdm(cimages_loader, desc="Inference"):
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)
            for number, label, c in zip(numbers, pred.cpu().tolist(), conf.cpu().tolist()):
                predictions.append((number, label))
                confidences.append(c)
    elapsed = time.time() - t0
    predictions.sort(key=lambda x: x[0])

    result_path = os.path.join(TEST_LOG_DIR, f"{RUN_DT}_{MODEL_TAG}_result.txt")
    with open(result_path, "w") as f:
        f.write("number, label\n")
        for number, label in predictions:
            f.write(f"{number}, {label:02d}\n")

    class_dist = {}
    for _, label in predictions:
        class_dist[label] = class_dist.get(label, 0) + 1

    meta_path = os.path.join(TEST_LOG_DIR, f"{RUN_DT}_{MODEL_TAG}_test_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "run_dt": RUN_DT, "model": MODEL_TAG, "pth": args.test,
            "num_images": len(predictions),
            "inference_time_total_sec": round(elapsed, 3),
            "inference_time_per_image_ms": round(elapsed / len(predictions) * 1000, 3),
            "confidence_mean": round(sum(confidences) / len(confidences), 4),
            "confidence_min": round(min(confidences), 4),
            "confidence_max": round(max(confidences), 4),
            "class_distribution": class_dist,
        }, f, indent=2)

    print(f"추론 완료: {len(predictions)}장 | {elapsed:.2f}s ({elapsed/len(predictions)*1000:.1f}ms/img)")
    print(f"결과: {result_path}")
    exit()

# ─────────────────────────────────────────
# 학습 모드
# ─────────────────────────────────────────
results = []
EPOCH_CSV = os.path.join(MODEL_LOG_DIR, f"{RUN_DT}_{MODEL_TAG}_epoch_results.csv")
epoch_csv_fields = ["experiment", "pretrained", "lr", "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "best_acc"]
with open(EPOCH_CSV, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=epoch_csv_fields).writeheader()

GLOBAL_JSONL = os.path.join(LOG_DIR, "experiments.jsonl")

for exp in cfg["experiments"]:
    lr = float(exp["lr"])
    weight_decay = float(exp.get("weight_decay", 0.05))
    label_smoothing = float(exp.get("label_smoothing", 0.1))

    print(f"\n{'='*60}")
    print(f"실험: {exp['name']}  |  pretrained={exp['pretrained']}  |  lr={lr}")
    print(f"weight_decay={weight_decay}  |  label_smoothing={label_smoothing}")
    print(f"ema_decay={EMA_DECAY}  |  drop_path_rate={DROP_PATH_RATE}")
    print(f"fusion_dim={FUSION_DIM}  |  bottleneck_dim={BOTTLENECK_DIM}  |  aux_loss_weight={AUX_LOSS_WEIGHT}")
    print(f"{'='*60}")

    model = AdaptiveStage34FusionConvNeXtV2(
        pretrained=exp["pretrained"], num_classes=NUM_CLASSES,
    ).to(device)
    param_count = report_parameter_count(model)

    # 🔥 Layer-wise LR (2번에서 이식)
    optimizer = torch.optim.AdamW([
        {"params": model.stage3_projection.parameters(), "lr": lr * 0.5},
        {"params": model.stage4_projection.parameters(), "lr": lr * 0.5},
        {"params": model.stage_gate.parameters(),        "lr": lr * 0.5},
        {"params": model.classifier.parameters(),        "lr": lr},
        {"params": model.stage3_aux_classifier.parameters(), "lr": lr},
        {"params": model.stage4_aux_classifier.parameters(), "lr": lr},
        {"params": model.backbone.parameters(),          "lr": lr * 0.1},
    ], weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 🔥 Warmup + CosineAnnealing (2번에서 이식)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_acc = 0.0
    start_epoch = 1
    ema_model = None
    ckpt_path = os.path.join(MODEL_LOG_DIR, f"{RUN_DT}_{MODEL_TAG}_{exp['name']}_checkpoint.pth")
    best_pth = os.path.join(MODEL_LOG_DIR, f"{RUN_DT}_{MODEL_TAG}_{exp['name']}_best.pth")

    if args.new:
        latest = None
    elif args.resume:
        latest = args.resume
    else:
        latest = find_latest_checkpoint(exp["name"])

    if latest:
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        if "ema_model" in ckpt and ckpt["ema_model"] is not None:
            ema_model = copy.deepcopy(model)
            ema_model.load_state_dict(ckpt["ema_model"])
            ema_model.eval()
        print(f"  → 체크포인트 로드: {latest} | epoch {start_epoch}부터 재개 (best {best_acc*100:.1f}%)")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, ema_model=ema_model
        )

        # 🔥 EMA 모델 첫 epoch 초기화 (2번과 동일)
        if ema_model is None:
            ema_model = copy.deepcopy(model)
            ema_model.eval()

        # 🔥 평가는 EMA 모델로 (2번과 동일)
        test_loss, test_acc = evaluate(ema_model, cifar_test_loader, criterion)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(ema_model.state_dict(), best_pth)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        }, ckpt_path)

        with open(EPOCH_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=epoch_csv_fields).writerow({
                "experiment": exp["name"], "pretrained": exp["pretrained"], "lr": lr,
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc * 100, 2),
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc * 100, 2),
                "best_acc": round(best_acc * 100, 2),
            })

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Test(CIFAR) Loss: {test_loss:.4f} Acc: {test_acc*100:.1f}% | "
            f"Best: {best_acc*100:.1f}%"
        )

    best_test_acc = round(best_acc * 100, 2)
    results.append({
        "experiment": exp["name"], "pretrained": exp["pretrained"],
        "lr": lr, "best_test_acc": best_test_acc,
    })
    print(f"→ {exp['name']} 최종 Best Accuracy: {best_test_acc}%")

    with open(GLOBAL_JSONL, "a") as f:
        json.dump({
            "run_dt": RUN_DT, "model": MODEL_TAG,
            "experiment": exp["name"], "pretrained": exp["pretrained"],
            "lr": lr, "epochs": EPOCHS,
            "fusion_out_indices": list(FUSION_OUT_INDICES),
            "fusion_dim": FUSION_DIM, "bottleneck_dim": BOTTLENECK_DIM,
            "dropout": DROPOUT, "aux_loss_weight": AUX_LOSS_WEIGHT,
            "ema_decay": EMA_DECAY, "drop_path_rate": DROP_PATH_RATE,
            "num_parameters": param_count,
            "best_test_acc": best_test_acc,
            "epoch_csv": EPOCH_CSV, "best_pth": best_pth, "checkpoint_pth": ckpt_path,
        }, f)
        f.write("\n")

# ─────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print("실험 결과 요약")
print(f"{'='*60}")
print(f"{'실험명':<25} {'Pretrained':<12} {'LR':<10} {'Best Acc'}")
print("-" * 60)
for r in sorted(results, key=lambda x: -x["best_test_acc"]):
    print(f"{r['experiment']:<25} {str(r['pretrained']):<12} {str(r['lr']):<10} {r['best_test_acc']}%")

summary_csv = os.path.join(MODEL_LOG_DIR, f"{RUN_DT}_{MODEL_TAG}_best.csv")
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["experiment", "pretrained", "lr", "best_test_acc"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n{summary_csv} 저장 완료")