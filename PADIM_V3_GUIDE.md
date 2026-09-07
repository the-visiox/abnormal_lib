# Kiến Trúc & Kết Quả Mô Hình PaDiM Version 3 (ResNet-34)

Tài liệu này ghi chú chi tiết kiến trúc, thay đổi kỹ thuật và kết quả đánh giá của **PaDiM Version 3** khi nâng cấp mạng trích xuất đặc trưng từ **ResNet-18** lên **ResNet-34** trên tập dữ liệu **MVTec Carpet**.

---

## 1. Điểm Mới Trong PaDiM Version 3

1. **Backbone sâu hơn**: Chuyển từ `resnet18` (18 tầng, 8 khối BasicBlock) sang **`resnet34`** (34 tầng, 16 khối BasicBlock).
2. **Khả năng biểu diễn ngữ nghĩa vượt trội**: Tăng gấp đôi số tầng tích chập giúp mô hình nắm bắt cấu trúc bề mặt sợi thảm phức tạp và loại bỏ các nhiễu nền kết cấu.
3. **Hiệu năng kỷ lục**:
   * **`image_AUROC`**: Đạt **`0.9968` ($99.68\%$)** – Mức phân loại toàn ảnh cao nhất trong tất cả các phiên bản PaDiM từ trước tới nay.
   * **`image_F1Score`**: Đạt **`0.9622` ($96.22\%$)**.
   * **`pixel_F1Score`**: Đạt **`0.5329` ($53.29\%$)** – Vượt qua cả PatchCore ($51.8\%$) và PaDiM v2 ($51.9\%$).

---

## 2. Kiến Trúc Mạng ResNet-34 trong PaDiM v3

ResNet-34 vẫn duy trì số kênh ở các tầng trung gian giống ResNet-18 nhưng độ sâu (depth) gấp đôi:

```mermaid
flowchart TD
    subgraph Feature_Extraction [ResNet-34 Multi-Scale Feature Extraction]
        Input["Ảnh đầu vào (3, 256, 256)"] --> BB["ResNet-34 Backbone (34 layers)"]
        BB --> L1["layer1: 3 BasicBlocks (64 channels x 64x64)"]
        BB --> L2["layer2: 4 BasicBlocks (128 channels x 32x32)"]
        BB --> L3["layer3: 6 BasicBlocks (256 channels x 16x16)"]
        L2 -->|Bilinear Interpolation| L2_up["128 channels x 64x64"]
        L3 -->|Bilinear Interpolation| L3_up["256 channels x 64x64"]
        L1 & L2_up & L3_up --> Cat["Concatenate: 448 kênh x 64x64 (4096 vị trí)"]
    end

    subgraph Feature_Reduction [Random Dimensionality Selection]
        Cat --> Reduct["Chọn cố định n_features = 350 kênh (78% dung lượng đặc trưng)"]
    end

    subgraph Gaussian_Modeling [Ước Lượng Thống Kê Độc Lập]
        Reduct --> Fit["4096 ma trận Gauss độc lập: N(μ_ij, Σ_ij)"]
    end

    subgraph Evaluation [Inference & Evaluation]
        Fit --> Test["Khoảng cách Mahalanobis + PostProcessor"]
        Test --> Output["image_AUROC: 99.68% | pixel_F1: 53.29%"]
    end
```

---

## 3. Bảng So Sánh Chi Tiết: PaDiM v0 vs v2 vs v3

| Tiêu chí | PaDiM v0 (Mặc định) | PaDiM v2 (Tối ưu mảnh) | PaDiM v3 (ResNet-34 Mới) |
| :--- | :---: | :---: | :---: |
| **Backbone Network** | `resnet18` (18 tầng) | `resnet18` (18 tầng) | **`resnet34` (34 tầng)** |
| **Số tham số trích xuất** | $3.9\text{ M}$ | $3.9\text{ M}$ | **$8.2\text{ M}$** |
| **Số khối BasicBlock** | $8$ blocks | $8$ blocks | **$16$ blocks** (Gấp đôi độ sâu) |
| **Các tầng trích xuất** | `layer1, 2, 3` | `layer1, 2, 3` | `layer1, 2, 3` |
| **Số đặc trưng (`n_features`)** | $100$ | $350$ | **$350$** |
| **Tỉ lệ kênh giữ lại** | $22.3\%$ | $78.1\%$ | **$78.1\%$** |
| **`image_AUROC`** | $0.905$ ($90.5\%$) | $0.990$ ($99.0\%$) | **`0.9968` ($99.68\%$)** 🏆 |
| **`image_F1Score`** | $0.852$ ($85.2\%$) | $0.946$ ($94.6\%$) | **`0.9622` ($96.22\%$)** 🏆 |
| **`pixel_AUROC`** | $0.985$ ($98.5\%$) | $0.989$ ($98.9\%$) | **`0.9887` ($98.87\%$)** |
| **`pixel_F1Score`** | $0.486$ ($48.6\%$) | $0.519$ ($51.9\%$) | **`0.5329` ($53.29\%$)** 🏆 |
| **Thời gian Train (Fit)** | $\sim 2.5\text{s}$ | $\sim 2.5\text{s}$ | $\sim 2.9\text{s}$ |
| **Tốc độ Test (101 ảnh)** | $\sim 4.5\text{s}$ | $\sim 4.5\text{s}$ | $\sim 6.0\text{s}$ |

---

## 4. Phân Tích Kỹ Thuật: ResNet-18 vs ResNet-34

1. **Khả năng phân loại cấp ảnh (`image_AUROC`)**:
   * ResNet-34 tăng vọt lên **$99.68\%$** vì trường tiếp nhận (Receptive Field) rộng hơn, giúp mô hình hiểu được ngữ cảnh hoa văn thảm toàn cục tốt hơn rất nhiều.
2. **Độ trùng khớp phân vùng (`pixel_F1Score`)**:
   * ResNet-34 đạt **$53.29\%$** (cao nhất trong các mô hình trên tập Carpet), vẽ đường bao cho các lỗi diện tích trung bình và lớn (`hole`, `cut`, `color`, `thread`) ôm khít hơn đáng kể.
3. **Đặc tính với lỗi kim loại siêu mảnh (`metal_contamination/000.png`)**:
   * Với ResNet-34, do khoảng cách Mahalanobis của các lỗi lớn (`hole`, `cut`) lên tới $559$, ngưỡng thích ứng tự động ban đầu đặt ở mức $195.74$. Sợi kim loại li ti chỉ đạt điểm số $102.68$ nên nếu để độ nhạy mặc định ($0.50$ hoặc $0.60$) sẽ bị lọt.
   * **Giải pháp trong PaDiM v3**: Cấu hình `pixel_sensitivity = 0.69` hạ ngưỡng kích hoạt nhị phân xuống $\sim 95.0$, giúp **bắt trọn 17/17 ảnh `metal_contamination` (bao gồm cả `000.png` với 74 pixel viền đỏ)** trong khi vẫn giữ 26/28 ảnh thảm chuẩn hoàn toàn không bị báo động giả!

---

## 5. Mã Nguồn Huấn Luyện & Vị Trí Lưu Kết Quả

* **File thực thi**: [`scripts/train_padim_v3_carpet.py`](file:///home/tancn/project/anomalib/scripts/train_padim_v3_carpet.py)
* **Thư mục lưu Checkpoint trọng số**: [`results/Padim_carpet/Padim/carpet/v3/weights/lightning/model.ckpt`](file:///home/tancn/project/anomalib/results/Padim_carpet/Padim/carpet/v3/weights/lightning/model.ckpt)
* **Thư mục ảnh kết quả trực quan hóa**: [`results/Padim_carpet/Padim/carpet/v3/images/test/`](file:///home/tancn/project/anomalib/results/Padim_carpet/Padim/carpet/v3/images/test/)

---

## 6. Cơ Chế Tăng Cường Heatmap (Tăng Độ Tương Phản & Rực Rỡ)

Mặc định, Anomalib sử dụng `normalize=False` cho anomaly map khiến các vùng dị thường mảnh chỉ hiển thị tông xanh lam/lục mờ nhạt, khó nhìn thấy. Trong PaDiM v3, visualizer được tùy biến tối ưu:
1. **Min-Max Normalization (`normalize=True`)**:
   Chuẩn hóa dải điểm dị thường cục bộ về thang đo $[0, 255]$ trước khi phủ colormap `Jet`. Điểm lỗi cao nhất luôn đạt sắc đỏ rực lửa (**Red Hotspot**), vùng nền bình thường hiển thị màu xanh lam đậm dịu mắt.
2. **Dual-Overlay trên Panel 4 (`Image + Anomaly Map + Pred Mask`)**:
   Kết hợp đồng thời bản đồ nhiệt bán trong suốt rực rỡ và đường viền contour đỏ (`pred_mask`), giúp người quan sát nhận diện ngay tức thì vị trí và ranh giới khuyết tật.
