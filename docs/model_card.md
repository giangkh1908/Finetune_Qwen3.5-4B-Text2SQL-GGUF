---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- text-to-sql
- sql
- unsloth
- qwen3.5
- reasoning
- cot
- gguf
- ollama
pipeline_tag: text-generation
---

# Qwen3.5-4B-Text2SQL: DeepSeek-Style Reasoning & Multi-Table Specialist (GGUF & LoRA)

Mô hình chuyên sâu về **Text-to-SQL phức tạp**, được fine-tune từ **`Qwen/Qwen3.5-4B`** bằng công nghệ **QLoRA 4-bit (Unsloth)** trên tập dữ liệu tuyển chọn **17.000 mẫu nâng cao** (kết hợp từ Gretel AI, Spider, và BIRD-Bench).

Mô hình hỗ trợ **Native Reasoning (`<think> ... </think>`)**, tự động suy luận cấu trúc bảng, liên kết khóa ngoại và điều kiện logic trước khi sinh câu lệnh SQL chuẩn SQLite.

---

## 🚀 Cách Chạy Nhanh Bằng Ollama (1 Dòng Lệnh)

Không cần tải file thủ công, Ollama hỗ trợ kéo trực tiếp từ Hugging Face:

```bash
ollama run hf.co/giangkh19/qwen3.5-4b-sql-gguf:Q4_K_M
```

> **Yêu cầu phần cứng**: Chạy siêu mượt trên máy tính cá nhân (chỉ cần **4GB VRAM GPU** như GTX 1650 hoặc **8GB RAM** nếu chạy CPU).

---

## 🎯 Cấu Trúc Prompt Chuẩn (ChatML Format)

```text
<|im_start|>system
You are a SQLite expert. Given the database schema, write the correct SQL query.

### DATABASE SCHEMA:
CREATE TABLE customers (id INT PRIMARY KEY, name TEXT, tier TEXT);
CREATE TABLE orders (order_id INT, customer_id INT, amount REAL, FOREIGN KEY(customer_id) REFERENCES customers(id));
<|im_end|>
<|im_start|>user
Find top 3 VIP customers with total spent over 500?
<|im_end|>
<|im_start|>assistant
<think>
1. Identify target tables: 'customers' and 'orders'.
2. Join condition: customers.id = orders.customer_id.
3. Filter conditions: tier = 'VIP'.
4. Aggregation: SUM(amount) grouped by customers.id having sum > 500.
5. Sort descending and limit to 3.
</think>
SELECT c.name, SUM(o.amount) AS total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE c.tier = 'VIP'
GROUP BY c.id, c.name
HAVING total_spent > 500
ORDER BY total_spent DESC
LIMIT 3;
<|im_end|>
```

---

## 📊 Đặc Điểm Tập Dữ Liệu Huấn Luyện (17.000 Mẫu Khó)

Đã loại bỏ 100% các câu đơn giản (1 bảng). Toàn bộ dữ liệu được chọn lọc khắt khe:
* **Multi-table JOINs (70%+)**: Liên kết khóa ngoại từ 2 đến 4 bảng lồng nhau.
* **Multi-conditions**: Tối thiểu 2 điều kiện lọc logic (`AND`/`OR`), xử lý ngày tháng (`strftime`), tính toán tỷ lệ.
* **Hàm nâng cao**: `GROUP BY`, `HAVING`, `CASE WHEN`, `UNION`, `EXCEPT`, `WINDOW functions` và `CTEs`.

---

## ⚙️ Cấu Hình Fine-Tuning (Unsloth QLoRA)

* **Base Model**: `Qwen/Qwen3.5-4B` (4-bit NF4 Quantization)
* **LoRA Parameters**: Rank $r = 16$, $lpha = 32$, Target 7 modules (`q, k, v, o, gate, up, down`)
* **Loss Masking**: `train_on_responses_only` (chỉ tính điểm phạt trên câu lệnh SQL, không phạt trên Schema)
* **Hardware**: Huấn luyện trên NVIDIA RTX 3080 Ti (12GB) với BF16 native trong ~2 giờ.
* **Final Loss**: Đạt mức **~0.35** (Loss cực kỳ đẹp, không overfitting).
