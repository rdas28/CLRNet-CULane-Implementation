# CLRerNet on CULane: Improving Confidence in Lane Detection with LaneIoU

This repository contains the implementation, evaluation, and analysis of **CLRerNet**, a state-of-the-art lane detection model, on the **CULane dataset** using the **LaneIoU metric** for improved confidence prediction.

Developed as part of a graduate-level Computer Vision project at the Illinois Institute of Technology.

## 🧠 Problem Statement

Lane detection is critical for autonomous driving systems, but **traditional models fail to produce reliable confidence scores**, especially on complex curved lanes.

### 🔥 Key Issues:
- Misalignment between predicted confidence and true geometric accuracy (IoU)
- Poor generalization on curves, night scenes, occlusions
- Existing metrics (like LineIoU) underperform on tilted lanes

---

## ✅ Our Solution

We implement **CLRerNet** and integrate **LaneIoU** into:
- Loss calculation
- Sample assignment (Dynamic-k strategy)
- Confidence supervision

This enhances model robustness and correlation between predicted confidence and actual lane quality.

---

## 🏗️ Model Architecture

- **Backbone:** DLA-34
- **Neck:** Feature Pyramid Network (FPN)
- **Head:** Modified proposal head for classification, regression, and confidence
- **Loss Function:** Includes LaneIoU-based confidence loss

---

## 📦 Dataset: CULane

- 133K frames @ 1640×590 resolution
- Scenarios: urban, curves, night, glare, occlusion
- Filtered down to 62,532 valid samples
- Lane masks from: `laneseg_label_w16/`

---

## 🧪 Implementation Details

| Component              | Details                             |
|------------------------|-------------------------------------|
| Framework              | MMDetection v3.0.0, MMCV v2.0.0     |
| Environment            | Docker-based training setup         |
| Optimizer              | AdamW, LR = 1e-4                    |
| Epochs                 | 2                                   |
| Batch Size             | 4 (GPU constrained)                 |
| Runtime Issues         | CPU training was extremely slow     |

---

## 🧮 LaneIoU: Custom Metric

A more accurate row-wise overlap metric for comparing predicted and ground truth lanes—especially effective on curves.

```python
def ComputeLaneIoU(predicted_lane, ground_truth_lane):
    overlap, valid_rows = 0, 0
    for y in range(image_height):
        x_pred, x_gt = predicted_lane[y], ground_truth_lane[y]
        if x_pred and x_gt are valid:
            iou_row = max(0, 1 - abs(x_pred - x_gt) / max_threshold)
            overlap += iou_row
            valid_rows += 1
    return overlap / valid_rows if valid_rows > 0 else 0
