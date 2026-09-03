# Hướng Dẫn Fine-Tuning Toàn Diện Với Unsloth & QLoRA

Tài liệu này cung cấp hướng dẫn chuyên sâu về kỹ thuật **QLoRA (Quantized Low-Rank Adaptation)**, lý giải cơ chế tăng tốc và tiết kiệm bộ nhớ của **Unsloth**, cấu hình siêu tham số (Hyperparameters) chuẩn cho Text-to-SQL và kỹ thuật sống còn **Loss Masking (`train_on_responses_only`)**.

---

## 1. Tại Sao Lại Là Unsloth? (Unsloth Deep Dive)

Khi fine-tune một mô hình LLM lớn, 2 rào cản lớn nhất là: **Hết bộ nhớ GPU (Out-Of-Memory - OOM)** và **Tốc độ train quá chậm**.

Unsloth giải quyết triệt để 2 vấn đề này bằng cách:
1. **Viết lại toàn bộ hàm lan truyền ngược (Manual Backprop Kernels)**: Thay vì dựa vào cơ chế tính đạo hàm tự động (Autograd) của PyTorch vốn lưu rất nhiều ma trận trung gian làm ngốn VRAM, Unsloth viết trực tiếp các kernel GPU bằng **OpenAI Triton**.
2. **Fast Cross-Entropy & Fast RoPE**: Gộp các phép tính ma trận rời rạc (Fused Operations) giúp giảm thiểu số lần đọc/ghi giữa chip GPU và bộ nhớ VRAM (VRAM Bandwidth).
3. **Tiết kiệm 80% VRAM & Tăng tốc 2-5x**: Bạn có thể fine-tune mô hình 3B hoặc 7B ngay trên một GPU bình dân chỉ có **8GB - 12GB VRAM** hoặc Google Colab T4 miễn phí.

---

## 2. Kỹ Thuật Sống Còn: Loss Masking (`train_on_responses_only`)

Trong bài toán Text-to-SQL, Prompt đầu vào bao gồm:
* Câu lệnh hệ thống (System Prompt)
* Toàn bộ cấu trúc Database (CREATE TABLE 5 - 15 bảng)
* Câu hỏi tự nhiên của người dùng

**Vấn đề**: Nếu tính điểm phạt (Loss) trên cả Prompt, mô hình sẽ tốn 80% tài nguyên để học thuộc bảng DDL và cách đặt câu hỏi, thay vì tập trung vào kỹ năng sinh SQL!

```text
Prompt (Schema + Câu hỏi)       ───►  Gán nhãn Label = -100  ───►  KHÔNG TÍNH LOSS
Response (Câu lệnh SQL sinh ra) ───►  Gán nhãn Token ID thật ───►  TÍNH LOSS & CẬP NHẬT TRỌNG SỐ
```

Trong Unsloth, việc này được thực hiện qua hàm `train_on_responses_only`:

```python
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)
```

* **Lợi ích**:
  * Giảm loss giả, mô hình hội tụ nhanh gấp đôi.
  * Tránh hiện tượng model bị "vẹt", sinh lặp lại chính câu hỏi của người dùng.

---

## 3. Cấu Hình QLoRA Chuẩn Cho Text-to-SQL

```python
from unsloth import FastLanguageModel

# 1. Load Base Model 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3.5-4B",                  # Base Model 4B thế hệ mới
    max_seq_length = 2048,                           # Đủ chứa 10-15 bảng DDL
    load_in_4bit = True,                             # NF4 Quantization siêu tiết kiệm VRAM
)

# 2. Inject LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                         # Rank của LoRA (16 hoặc 32)
    lora_alpha = 32,                # Hệ số tỉ lệ alpha (thường gấp đôi rank)
    lora_dropout = 0,               # Đặt 0 để tối ưu tốc độ Unsloth
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],                              # Tác động lên cả Attention và MLP
    use_gradient_checkpointing = "unsloth", # Tiết kiệm thêm 30% VRAM
    random_state = 3407,
)
```

### Giải thích các siêu tham số LoRA:
* **Rank ($r$)**: Số chiều của ma trận nén. Với Text-to-SQL, $r=16$ là mức lý tưởng. Nếu tăng lên $r=64$, mô hình học chi tiết hơn nhưng dễ bị overfitting vào các database quen thuộc.
* **Alpha ($\alpha$)**: Trọng số điều chỉnh mức độ ảnh hưởng của LoRA. Tỷ lệ vàng là $\alpha / r = 2$.
* **Target Modules**: Bắt buộc phải gắn LoRA vào toàn bộ 7 khối (`q, k, v, o, gate, up, down`). Nếu chỉ gắn vào `q_proj` và `v_proj` như LoRA đời cũ, model sẽ không đủ khả năng học các logic JOIN phức tạp.

---

## 4. Bảng Siêu Tham Số Huấn Luyện (Hyperparameter Cookbook)

| Siêu tham số | Giá trị đề xuất | Ý nghĩa & Lý do lựa chọn |
| :--- | :--- | :--- |
| **Learning Rate** | `1.5e-4` đến `2e-4` | Tốc độ học tối ưu cho QLoRA 4B (không quá lớn gây quên tri thức gốc) |
| **LR Scheduler** | `cosine` | Giảm tốc độ học từ từ về 0 theo hình sin, giúp hội tụ mượt mà |
| **Warmup Ratio** | `0.05` (5% tổng steps) | Khởi động nhẹ nhàng để adapter ổn định trước khi tăng tốc |
| **Batch Size per device** | `2` | Phù hợp với context length 2048 tokens trên GPU cá nhân |
| **Gradient Accumulation** | `8` | Tích lũy gradient để tạo **Effective Batch Size = 16** |
| **Weight Decay** | `0.01` | Điều chuẩn $L_2$ để tránh overfitting |
| **Max Steps / Epochs** | `1` đến `2` epochs | 19.000 mẫu chỉ cần train 1-2 epochs là đạt điểm cực đại |
| **Optimizers** | `adamw_8bit` | Tối ưu hóa AdamW dạng 8-bit tiết kiệm VRAM tối đa |

---

## 5. Bảng Ước Tính Dung Lượng VRAM & Phần Cứng

| Cấu Hình Model | Chiều Dài Context | Batch Size | Bộ Nhớ VRAM Cần | Phần Cứng Đáp Ứng |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3.5-4B (4-bit)** | 2048 tokens | 2 | **~5.8 GB** | RTX 3060 (Laptop/Desktop 6GB/12GB), RTX 4060 (8GB) |
| **Qwen3.5-4B (4-bit)** | 4096 tokens | 2 | **~6.8 GB** | RTX 3060 (12GB), RTX 4060 Ti, Colab T4 |
| **Qwen3.5-4B (16-bit Full)** | 2048 tokens | 2 | **~14.5 GB** | Cần RTX 3090 / 4090 / A100 |

> **Khuyến nghị cho dự án**: Huấn luyện **`Qwen/Qwen3.5-4B` bản 4-bit NF4** với context **2048 tokens**, batch size 2, gradient accumulation 8. Thời gian train trên 1 GPU thông thường mất khoảng **70 - 100 phút**, mang lại khả năng suy luận vượt trội so với các model 1.5B/3B!
