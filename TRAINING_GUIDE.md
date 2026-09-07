# Hướng Dẫn Huấn Luyện Mô Hình Anomaly Detection (Anomalib)

Tài liệu này tổng hợp toàn bộ quy trình từ cấu trúc dữ liệu, mã nguồn huấn luyện, tối ưu độ nhạy phát hiện khuyết tật đến cách phân tích kết quả trên bài toán phát hiện bất thường (ví dụ cụ thể trên dataset **Carpet** với mô hình **PatchCore**).

---

## 1. Cấu Trúc Thư Mục Chuẩn

### 1.1. Cấu trúc tổng thể của Workspace
```text
anomalib/
├── datasets/                           # Thư mục chứa các bộ dữ liệu
│   └── carpet/                         # Bộ dữ liệu mẫu (MVTec AD Carpet)
├── results/                            # Thư mục lưu kết quả tự động sau khi train
│   └── Patchcore/
│       └── carpet/
│           ├── latest -> v3/           # Shortcut trỏ đến phiên bản mới nhất
│           └── v3/                     # Phiên bản chạy lần 3
│               ├── images/test/        # Ảnh trực quan hóa kết quả test
│               └── weights/lightning/  # Checkpoint trọng số mô hình (model.ckpt)
├── src/anomalib/                       # Mã nguồn thư viện Anomalib
├── scripts/                            # Chứa các script huấn luyện
│   ├── train_padim_carpet.py
│   ├── train_patchcore_carpet.py
│   └── train_patchcore_mvtec.py
├── PADIM_V2_GUIDE.md                   # Tài liệu chi tiết PaDiM Version 2
└── TRAINING_GUIDE.md                   # Tài liệu hướng dẫn này
```

---

### 1.2. Cấu trúc thư mục Dataset (`datasets/carpet`)
Anomalib sử dụng cơ chế `Folder DataModule` chuẩn cho bài toán học một lớp (One-Class Learning / Unsupervised). Dữ liệu được tổ chức như sau:

```text
datasets/carpet/
├── train/
│   └── good/                       # 280 ảnh chuẩn (không có lỗi) dùng để huấn luyện
├── test/
│   ├── good/                       # 28 ảnh bình thường dùng để kiểm tra False Alarm
│   ├── color/                      # Ảnh lỗi đốm màu (19 ảnh)
│   ├── cut/                        # Ảnh vết rách, cắt (17 ảnh)
│   ├── hole/                       # Ảnh lỗ thủng (17 ảnh)
│   ├── metal_contamination/        # Ảnh dính mạt/vết kim loại (17 ảnh)
│   └── thread/                     # Ảnh bung sợi chỉ dệt (19 ảnh)
└── ground_truth/                   # Mặt nạ nhãn (Binary Mask: trắng = lỗi, đen = nền)
    ├── color/
    ├── cut/
    ├── hole/
    ├── metal_contamination/
    └── thread/
```

> **Quy tắc quan trọng**:
> - Tên file trong thư mục `ground_truth/<loại_lỗi>/` phải tương ứng với tên file trong `test/<loại_lỗi>/` (ví dụ: `test/cut/000.png` có mask là `ground_truth/cut/000_mask.png`).
> - Tập `train/good/` **chỉ** chứa sản phẩm đạt chuẩn.

---

## 2. Thiết Lập Môi Trường (Environment)

Dự án sử dụng môi trường Python quản lý qua Conda (`reid`) đã được cài đặt sẵn PyTorch và CUDA:

```bash
# Kích hoạt môi trường
conda activate reid

# Hoặc chạy trực tiếp qua đường dẫn binary:
python scripts/train_patchcore_carpet.py
```

---

## 3. Chi Tiết File Huấn Luyện (`scripts/train_patchcore_carpet.py`)

Nội dung mã nguồn tối ưu được tổ chức thành 3 bước logic:

```python
"""Train PatchCore trên dataset carpet."""

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor

# ── 1. Cấu hình DataModule ───────────────────────────────────────────────────
# Liệt kê tất cả các loại lỗi cần test và thư mục ground_truth tương ứng
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
    root="./datasets/carpet",
    normal_dir="train/good",          # Tập train bình thường
    abnormal_dir=DEFECT_TYPES,         # Các nhóm lỗi
    normal_test_dir="test/good",       # Tập test bình thường
    mask_dir=MASK_DIRS,                # Mặt nạ ground truth
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=4,
)

# ── 2. Cấu hình Model & PostProcessor ────────────────────────────────────────
# Tinh chỉnh độ nhạy: Mặc định pixel_sensitivity=0.5 sẽ đẩy ngưỡng pixel lên cao (~42.7),
# khiến các lỗi mảnh (như sợi chỉ bung, lỗ kim nhỏ) không vẽ được viền đỏ.
# Thiết lập pixel_sensitivity=0.65 giúp bắt trọn 93.3% lỗi mà False Positive vẫn là 0.
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

# ── 3. Engine Huấn Luyện & Đánh Giá ──────────────────────────────────────────
engine = Engine(
    max_epochs=1,
    accelerator="auto",
)

# Huấn luyện (trích xuất đặc trưng & xây dựng Memory Bank Coreset)
print("=== Bắt đầu train PatchCore trên dataset carpet ===")
engine.fit(model=model, datamodule=datamodule)

# Đánh giá & xuất ảnh trực quan hóa
print("\n=== Đánh giá trên test set ===")
results = engine.test(model=model, datamodule=datamodule)
print("\nKết quả:", results)
```

---

## 4. Các Chỉ Số Đánh Giá (Evaluation Metrics)

Khi kết thúc quá trình test, Anomalib tự động tính toán bảng chỉ số:

| Chỉ số | Giá trị đạt được | Ý nghĩa |
| :--- | :--- | :--- |
| **`image_AUROC`** | **`0.978` (97.8%)** | Đo khả năng xếp hạng phân biệt giữa ảnh tốt và ảnh lỗi ở mọi ngưỡng (threshold-independent). |
| **`image_F1Score`** | **`0.946` (94.6%)** | Điểm F1 cân bằng giữa Precision và Recall ở mức toàn ảnh tại ngưỡng tối ưu. |
| **`pixel_AUROC`** | **`0.991` (99.1%)** | Đo độ chuẩn xác của bản đồ nhiệt (Anomaly Map) trong việc định vị từng pixel khuyết tật. |
| **`pixel_F1Score`** | **`0.518` (51.8%)** | Độ trùng khớp hình dạng hình học giữa vùng viền dự đoán (`Pred Mask`) và `Ground Truth`. |

---

## 5. Cấu Trúc File Xuất Kết Quả Trực Quan (`results/`)

Sau khi chạy xong `engine.test()`, hệ thống tự động sinh thư mục lưu trữ:

```text
results/Patchcore/carpet/latest/
├── images/
│   └── test/
│       ├── good/                     # Ảnh bình thường (kiểm tra báo động giả)
│       │   └── 001.png
│       ├── color/                    # Ảnh trực quan hóa lỗi đốm màu
│       ├── cut/                      # Ảnh trực quan hóa lỗi rách
│       ├── hole/                     # Ảnh trực quan hóa lỗ thủng
│       ├── metal_contamination/      # Ảnh trực quan hóa kim loại
│       └── thread/                   # Ảnh trực quan hóa bung sợi chỉ
└── weights/
    └── lightning/
        └── model.ckpt                # Trọng số mô hình và trạng thái post-processor
```

### Cách đọc ảnh trực quan hóa (gồm 4 khung ghép ngang):
Mỗi file ảnh kết quả (ví dụ `results/Patchcore/carpet/latest/images/test/thread/002.png`) được ghép bởi 4 ô:
1. **`Image`**: Ảnh đầu vào thực tế từ camera/tập test.
2. **`Gt Mask`**: Mặt nạ nhãn thực tế (Ground Truth) do chuyên gia gắn nhãn.
3. **`Image + Anomaly Map`**: Bản đồ nhiệt phủ lên ảnh gốc (xanh lam = bình thường, vàng/đỏ = mức độ dị thường cao).
4. **`Image + Pred Mask`**: Đường bao viền màu đỏ khoanh vùng khuyết tật sau khi áp dụng ngưỡng nhị phân.

---

## 6. Kinh Nghiệm Thực Tế & Xử Lý Vấn Đề (Troubleshooting)

### 6.1. Tại sao nhiều hình không hiện viền đỏ ở `Pred Mask`?
- **Nguyên nhân**: Mặc định `F1AdaptiveThreshold` tự động tìm ngưỡng tối đa hóa F1 pixel. Vì số pixel lỗi trong toàn bộ tập chỉ chiếm $< 0.5\%$, thuật toán nâng ngưỡng pixel lên rất cao ($> 42.7$) để tránh tuyệt đối báo lỗi giả. Kết quả là các lỗi mảnh/mờ (sợi chỉ, lỗ nhỏ) dù bản đồ nhiệt đã phát hiện được nhưng không vượt qua được ngưỡng cắt.
- **Khắc phục**: Tăng `pixel_sensitivity` trong `PostProcessor(pixel_sensitivity=0.65)` để hạ ngưỡng cắt một cách có kiểm soát. Tỉ lệ phát hiện lỗi tăng từ **80% (36/45)** lên **93.3% (42/45)** mà không sinh báo động giả.

### 6.2. Tại sao không nên thêm `layer1` vào PatchCore?
- `layer1` có kích thước không gian rất lớn ($64 \times 64 = 4096$ patches/ảnh). Với 280 ảnh train, số lượng embedding tích lũy vượt quá hàng triệu vector, khi gom vào memory bank sẽ gây lỗi tràn bộ nhớ VRAM (**CUDA Out Of Memory > 16GB**).
- **Khuyến nghị**: Sử dụng chuẩn `layers=["layer2", "layer3"]` của PatchCore kết hợp tinh chỉnh `PostProcessor`.

### 6.3. Tái sử dụng Checkpoint đã train mà không cần train lại
Nếu chỉ muốn test lại hoặc suy luận (predict) với ngưỡng khác, không cần gọi `engine.fit()`, chỉ cần truyền đường dẫn checkpoint:

```python
engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path="results/Patchcore/carpet/latest/weights/lightning/model.ckpt"
)
```
