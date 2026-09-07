from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 1. Datamodule ────────────────────────────────────────────────────────────
# Capsule có 5 loại lỗi — truyền tất cả vào abnormal_dir

DEFECT_TYPES = [
    "test/crack",
    "test/faulty_imprint",
    "test/poke",
    "test/scratch",
    "test/squeeze",
]

MASK_DIRS = [
    "ground_truth/crack",
    "ground_truth/faulty_imprint",
    "ground_truth/poke",
    "ground_truth/scratch",
    "ground_truth/squeeze",
]

datamodule = Folder(
    name="capsule",
    root=PROJECT_ROOT / "datasets/capsule",
    normal_dir="train/good",
    abnormal_dir=DEFECT_TYPES,
    normal_test_dir="test/good",
    mask_dir=MASK_DIRS,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=4,
)

# ── 2. Model ─────────────────────────────────────────────────────────────────

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=9,
)

# ── 3. Engine & Train ────────────────────────────────────────────────────────

engine = Engine(
    max_epochs=1,
    accelerator="auto",
    default_root_dir=PROJECT_ROOT / "results/Patchcore_capsule",
)

print("=== Bắt đầu train PatchCore trên dataset capsule ===")
engine.fit(model=model, datamodule=datamodule)

print("\n=== Đánh giá trên test set ===")
results = engine.test(model=model, datamodule=datamodule)
print("\nKết quả:", results)
