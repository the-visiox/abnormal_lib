from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 1. Cấu hình Datamodule ───────────────────────────────────────────────────
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
    val_split_mode="same_as_test",  # Đảm bảo kiểm tra toàn bộ ảnh test (không bị mất ảnh)
    train_batch_size=16,
    eval_batch_size=16,
    num_workers=4,
)

# ── 2. Cấu hình PaDiM Model & PostProcessor ──────────────────────────────────
# - n_features=350: Tăng từ 100 lên 350 (giữ lại 78% đặc trưng của 3 layers)
#   để không bỏ sót các tín hiệu ánh kim cực mảnh của sợi kim loại (metal_contamination).
# - pixel_sensitivity=0.60: Đủ nhạy để kích hoạt viền đỏ cho các khuyết tật siêu nhỏ.
post_processor = PostProcessor(
    pixel_sensitivity=0.60,
    image_sensitivity=0.52,
)

model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
    n_features=350,
    post_processor=post_processor,
)

# ── 3. Engine & Train ────────────────────────────────────────────────────────
# Lưu ý: PaDiM là mô hình thống kê phân phối Gauss (tính mean và covariance trực tiếp),
# không sử dụng hàm mất mát (loss) và lan truyền ngược (gradient descent).
# Do đó bản chất thuật toán chỉ cần 1 epoch duyệt qua tập dữ liệu để tính ma trận hiệp phương sai.
engine = Engine(
    accelerator="auto",
    default_root_dir=PROJECT_ROOT / "results/Padim_carpet",
)

print("=== Huấn luyện PaDiM Version 2 trên Carpet ===")
engine.fit(model=model, datamodule=datamodule)

print("\n=== Đánh giá trên test set ===")
results = engine.test(model=model, datamodule=datamodule)
print("\nKết quả PaDiM v2:", results)
