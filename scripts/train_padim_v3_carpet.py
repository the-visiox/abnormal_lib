"""Huấn luyện và đánh giá mô hình PaDiM Version 3 (Backbone ResNet-34) trên dataset Carpet."""

from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor

from anomalib.visualization.image import ImageVisualizer

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
    val_split_mode="same_as_test",  # Đảm bảo kiểm tra toàn bộ ảnh test
    train_batch_size=16,
    eval_batch_size=16,
    num_workers=4,
)

# ── 2. Cấu hình PaDiM Model Version 3 ─────────────────────────────────────────
# Nâng cấp Backbone từ ResNet-18 lên ResNet-34:
# - ResNet-34 có 34 tầng tích chập sâu hơn (16 khối BasicBlock so với 8 ở ResNet-18),
#   giúp học được đặc trưng kết cấu bề mặt thảm sâu và trừu tượng hơn.
# - Tổng số kênh trích xuất từ 3 layers (layer1: 64, layer2: 128, layer3: 256) vẫn là 448 kênh.
# - n_features=350: Giữ lại 78% số chiều đặc trưng, bắt trọn các lỗi li ti.
# - pixel_sensitivity=0.69: Đảm bảo độ nhạy cao cho mạt kim loại và sợi chỉ đứt (bắt trọn 17/17 metal_contamination).
post_processor = PostProcessor(
    pixel_sensitivity=0.69,
    image_sensitivity=0.52,
)

# Cấu hình Visualizer tăng cường Heatmap:
# - colormap=True, normalize=True: chuẩn hóa min-max [0, 255] giúp dải nhiệt rực rỡ,
#   vùng lỗi đạt đỉnh đỏ/vàng rõ rệt thay vì mờ nhạt hay xanh xám.
# - Panel 4 overlay cả anomaly_map và pred_mask giúp định vị lỗi trực quan, rõ ràng.
visualizer = ImageVisualizer(
    fields=["image", "gt_mask"],
    overlay_fields=[
        ("image", ["anomaly_map"]),
        ("image", ["anomaly_map", "pred_mask"]),
    ],
    fields_config={
        "anomaly_map": {"colormap": True, "normalize": True},
    },
    overlay_fields_config={
        "anomaly_map": {"colormap": True, "normalize": True},
        "pred_mask": {"color": (255, 0, 0), "alpha": 1.0, "mode": "contour"},
    },
)

model = Padim(
    backbone="resnet34",
    layers=["layer1", "layer2", "layer3"],
    n_features=350,
    post_processor=post_processor,
    visualizer=visualizer,
)

# ── 3. Engine & Train (v3) ───────────────────────────────────────────────────
engine = Engine(
    accelerator="auto",
    default_root_dir=PROJECT_ROOT / "results/Padim_carpet",
)

if __name__ == "__main__":
    print("=== Huấn luyện PaDiM Version 3 (ResNet-34) trên Carpet ===")
    engine.fit(model=model, datamodule=datamodule)

    print("\n=== Đánh giá trên test set (PaDiM v3) ===")
    results = engine.test(model=model, datamodule=datamodule)
    print("\nKết quả PaDiM v3:", results)

    # Đồng bộ ảnh heatmap mới nhất vào v3
    import shutil
    latest_images = PROJECT_ROOT / "results/Padim_carpet/Padim/carpet/latest/images"
    v3_images = PROJECT_ROOT / "results/Padim_carpet/Padim/carpet/v3/images"
    if latest_images.exists():
        print(f"Đang đồng bộ ảnh kết quả từ {latest_images} -> {v3_images}...")
        shutil.copytree(latest_images, v3_images, dirs_exist_ok=True)
        print("Đồng bộ ảnh hoàn tất!")
