from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 1. Datamodule ────────────────────────────────────────────────────────────
# Carpet có 5 loại lỗi — truyền tất cả vào abnormal_dir

DEFECT_TYPES = [
    "test/color",
    "test/cut",
    "test/hole",
    "test/metal_contamination",
    "test/thread",
]

MASK_DIRS = [
    "ground_truth/color",
    "ground_truth/cut",
    "ground_truth/hole",
    "ground_truth/metal_contamination",
    "ground_truth/thread",
]

datamodule = Folder(
    name="carpet",
    root=PROJECT_ROOT / "datasets/carpet",
    normal_dir="train/good",
    abnormal_dir=DEFECT_TYPES,
    normal_test_dir="test/good",
    mask_dir=MASK_DIRS,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=4,
)

# ── 2. Model & Post-processing ───────────────────────────────────────────────
# Tăng pixel_sensitivity lên 0.65 để phát hiện các khuyết tật nhỏ/mảnh (chỉ, lỗ kim, đốm mờ)
# mà không làm tăng báo động giả (False Positive = 0).
post_processor = PostProcessor(
    pixel_sensitivity=0.65,
    image_sensitivity=0.55,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=9,
    post_processor=post_processor,
)

# ── 3. Engine & Train ────────────────────────────────────────────────────────

engine = Engine(
    max_epochs=1,
    accelerator="auto",
)

print("=== Bắt đầu train PatchCore trên dataset carpet ===")
engine.fit(model=model, datamodule=datamodule)

print("\n=== Đánh giá trên test set ===")
results = engine.test(model=model, datamodule=datamodule)
print("\nKết quả:", results)
