# 验证准确率 ≥ 90% 的 Checkpoint 清单

根据 `train_log.csv`，从 **epoch 35** 起 `val_acc` 已 ≥ 0.90，之后所有保存的 checkpoint 都高于 90%。

## 一、目录里有的、且对应 epoch 时 val_acc ≥ 90% 的文件

以下文件在 `outputs_stage3_long/checkpoints/` 中**存在**，且该 epoch 的验证准确率 ≥ 90%：

| 文件 | epoch | val_acc (约) | 说明 |
|------|-------|--------------|------|
| **ep0592.pth** | 592 | **95.66%** | 当前最高，推荐 |
| ep0950.pth | 950 | 94.71% | |
| ep0900.pth | 900 | ≥90% | |
| ep0850.pth | 850 | ≥90% | |
| ep0800.pth | 800 | ≥90% | |
| ep0750.pth | 750 | ≥90% | |
| ep0700.pth | 700 | ≥90% | |
| ep0650.pth | 650 | ≥90% | |
| ep0600.pth | 600 | ≥90% | |
| ep0582.pth | 582 | ≥90% | |
| ep0591.pth | 591 | ≥90% | |
| ep0558.pth | 558 | ≥90% | |
| ep0557.pth | 557 | ≥90% | |
| ep0550.pth | 550 | ≥90% | |
| ep0532.pth | 532 | ≥90% | |
| ep0503.pth | 503 | ≥90% | |
| **ep0500.pth** | 500 | **95.12%** | |
| ep0450.pth | 450 | ≥90% | |
| **ep0400.pth** | 400 | **94.68%** | |
| ep0381.pth | 381 | ≥90% | |
| ep0350.pth | 350 | ≥90% | |
| **ep0300.pth** | 300 | **94.58%** | |
| ep0260.pth | 260 | ≥90% | |
| ep0259.pth | 259 | ≥90% | |
| ep0250.pth | 250 | ≥90% | |
| ep0241.pth | 241 | ≥90% | |
| ep0219.pth | 219 | ≥90% | |
| **ep0200.pth** | 200 | **94.14%** | |
| ep0162.pth | 162 | ≥90% | |
| ep0161.pth | 161 | ≥90% | |
| **ep0150.pth** | 150 | **94.48%** (best_acc) | |
| ep0101.pth | 101 | ≥90% | |
| **ep0100.pth** | 100 | **94.17%** | |
| ep0099.pth | 99 | ≥90% | |
| ep0095.pth | 95 | ≥90% | |
| ep0090.pth | 90 | ≥90% | |
| ep0086.pth | 86 | ≥90% | |
| ep0085.pth | 85 | ≥90% | |
| ep0083.pth | 83 | ≥90% | |
| ep0079.pth | 79 | ≥90% | |
| ep0075.pth | 75 | ≥90% | |
| ep0074.pth | 74 | ≥90% | |
| ep0073.pth | 73 | ≥90% | |
| ep0071.pth | 71 | ≥90% | |
| ep0069.pth | 69 | ≥90% | |
| ep0067.pth | 67 | ≥90% | |
| ep0065.pth | 65 | ≥90% | |
| ep0064.pth | 64 | ≥90% | |
| ep0063.pth | 63 | ≥90% | |
| ep0062.pth | 62 | ≥90% | |
| ep0061.pth | 61 | ≥90% | |
| ep0060.pth | 60 | ≥90% | |
| ep0057.pth | 57 | ≥90% | |
| ep0056.pth | 56 | ≥90% | |
| ep0055.pth | 55 | ≥90% | |
| ep0054.pth | 54 | ≥90% | |
| ep0052.pth | 52 | ≥90% | |
| ep0050.pth | 50 | 90.55% | |
| ep0048.pth | 48 | ≥90% | |
| ep0047.pth | 47 | ≥90% | |
| ep0039.pth | 39 | ≥90% | |
| ep0038.pth | 38 | ≥90% | |
| ep0037.pth | 37 | ≥90% | |
| ep0036.pth | 36 | ≥90% | |
| **ep0035.pth** | 35 | **90.07%** | 第一个 ≥90% 的 epoch |
| ep0033.pth | 33 | &lt;90% | 不满足 |
| ep0030.pth | 30 | &lt;90% | 不满足 |
| **latest.pth** | (最后一轮) | ≥90% | 当前训练最新，等价于最后一轮保存 |

**结论：除 ep0000～ep0030、ep0033 外，其余所有 `epXXXX.pth` 以及 `latest.pth` 都是「高于 90%」的，直接用任意一个即可。**

---

## 二、推荐用法（只想要一个）

- **要最高准确率**：用 **`ep0592.pth`**（val_acc ≈ 95.66%）。
- **要“只提一个推理用 .pth”**：在 `ViT` 目录下执行：
  ```bash
  python extract_best_checkpoint.py --epoch 592 --out outputs_stage3_long/best_ep592.pth
  ```
  会从 `ep0592.pth` 里抽出 EMA/model 的 state_dict，保存为 `best_ep592.pth`，推理时：
  ```python
  state_dict = torch.load("outputs_stage3_long/best_ep592.pth", map_location="cpu")
  model.load_state_dict(state_dict)
  ```

---

## 三、本地查看完整表格（含每 epoch 的 val_acc）

在项目里进入 `ViT` 目录后执行：

```bash
python list_checkpoints_above_90.py
```

会按 `train_log.csv` 列出所有「val_acc ≥ 90% 且文件存在」的 epoch 及对应路径。
