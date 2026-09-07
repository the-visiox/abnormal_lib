# Training Scripts

Thư mục chứa các script huấn luyện mẫu cho các mô hình phát hiện dị thường (Anomaly Detection) trên các tập dữ liệu thực tế.

## Danh sách Scripts

| Script | Mô hình | Dataset | Mô tả |
| :--- | :--- | :--- | :--- |
| [`train_padim_carpet.py`](file:///home/tancn/project/anomalib/scripts/train_padim_carpet.py) | **PaDiM v2** (ResNet-18) | MVTec Carpet | PaDiM Version 2 tối ưu `n_features=350`, `pixel_sensitivity=0.60`, bắt trọn 45/45 khuyết tật (kể cả sợi kim loại cực mảnh). |
| [`train_padim_v3_carpet.py`](file:///home/tancn/project/anomalib/scripts/train_padim_v3_carpet.py) | **PaDiM v3** (ResNet-34) | MVTec Carpet | PaDiM Version 3 với mạng ResNet-34 sâu gấp đôi, lập kỷ lục `image_AUROC = 99.68%`, `pixel_F1 = 53.29%`. |
| [`train_patchcore_carpet.py`](file:///home/tancn/project/anomalib/scripts/train_patchcore_carpet.py) | **PatchCore** (Wide-ResNet-50) | MVTec Carpet | PatchCore với cấu hình `pixel_sensitivity=0.65` cải thiện tỉ lệ bắt lỗi cho các khuyết tật dạng sợi, lỗ kim. |
| [`train_patchcore_mvtec.py`](file:///home/tancn/project/anomalib/scripts/train_patchcore_mvtec.py) | **PatchCore** (Wide-ResNet-50) | MVTec Capsule | Huấn luyện PatchCore trên danh mục đối tượng Capsule của MVTec AD. |

## Hướng dẫn chạy

Từ thư mục gốc của project, thực thi lệnh với môi trường Python:

```bash
# Huấn luyện PaDiM v2 trên Carpet
python scripts/train_padim_carpet.py

# Huấn luyện PatchCore trên Carpet
python scripts/train_patchcore_carpet.py

# Huấn luyện PatchCore trên Capsule
python scripts/train_patchcore_mvtec.py
```
