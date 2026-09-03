# DeepSeek-Style Text-to-SQL: Distillation & QLoRA Fine-tuning với Unsloth

Dự án nghiên cứu và thực hành end-to-end về **Knowledge Distillation** và **Fine-tuning LLM** chuyên sâu cho bài toán **Text-to-SQL phức tạp** (kết hợp nhiều bảng - Multi-table JOINs, nhiều điều kiện lọc lồng nhau - Multi-conditions, Aggregations, Window Functions và Subqueries).

---

## 🎯 Mục Tiêu Dự Án

1. **Hiểu sâu về Distillation trong kỷ nguyên LLM**:
   - Khám phá cách các mô hình lớn (Teacher: GPT-4o, Claude 3.5 Sonnet, Qwen-2.5-72B) chắt lọc tri thức và khả năng suy luận (Chain-of-Thought) sang mô hình nhỏ (Student: 1.5B - 3B).
   - Tận dụng các tập dữ liệu tổng hợp (Synthetic Datasets) mã nguồn mở chất lượng cao không tốn chi phí gọi API.
2. **Làm chủ Unsloth & QLoRA**:
   - Huấn luyện mô hình với 4-bit quantization, tiết kiệm tới 80% VRAM và tăng tốc 2-5x.
   - Nắm vững cơ chế **Loss Masking** (	rain_on_responses_only): Chỉ tính loss trên câu lệnh SQL sinh ra, giữ nguyên sự chú ý (Attention) vào Database Schema và câu hỏi mà không làm lãng phí năng lực mô hình.
3. **Data Harmonization & Anti-Leakage**:
   - Hợp nhất 3 nguồn dữ liệu chuẩn quốc tế: **Gretel Synthetic**, **Spider (Context)**, và **BIRD-Bench**.
   - Phân tách tập Train / Test theo **Database ID** (Zero-shot Database split) để đảm bảo mô hình có khả năng tổng quát hóa trên các cơ sở dữ liệu hoàn toàn mới trong thực tế.
4. **Đánh giá khách quan (Execution Accuracy)**:
   - Đo lường độ chính xác thực thi (Execution Accuracy - EX) bằng cách chạy câu lệnh SQL trực tiếp trên cơ sở dữ liệu SQLite thật thay vì chỉ so khớp chuỗi ký tự (Exact Match - EM).
5. **Đóng gói & Phục vụ (Serving)**:
   - Hợp nhất LoRA weights, xuất ra định dạng **GGUF** và chạy offline cục bộ với **Ollama** / **vLLM**.

---

## 📂 Cấu Trúc Dự Án & Tài Liệu Chi Tiết

```text
D:/FinetuneV1/
├── README.md                           # Giới thiệu tổng quan và hướng dẫn bắt đầu
├── requirements.txt                    # Danh sách thư viện cho Server GPU
├── download_data.py                    # Script tải tự động 3 tập dữ liệu thô
├── configs/
│   └── training_config.yaml            # Cấu hình siêu tham số tập trung
├── data/
│   ├── processed/                      # Dữ liệu huấn luyện đã lọc sạch (train.jsonl, val.jsonl)
│   └── benchmark/                      # Dữ liệu benchmark độc lập (test_benchmark.jsonl, sqlite_dbs/)
├── notebooks/
│   └── text2sql_qwen3_5_4b_colab.ipynb # Notebook 1-Click chạy trên Google Colab (T4/L4)
├── src/                                # Source code dùng chung (data_processor, trainer, evaluator)
├── scripts/                            # Scripts dòng lệnh cho GPU Server (01_prepare, 02_train, 03_eval, 04_export)
├── docs/                               # Bộ tài liệu kỹ thuật chuyên sâu
│   ├── architecture.md                 # Kiến trúc tổng thể & Qwen3.5-4B (4-bit NF4)
│   ├── distillation_guide.md           # Cẩm nang Knowledge & Reasoning Distillation
│   ├── data_pipeline.md                # Quy trình lọc câu khó, ChatML & Chống data leakage
│   ├── finetuning_unsloth.md           # Hướng dẫn cấu hình QLoRA và tối ưu Unsloth
│   ├── evaluation_benchmark.md         # Quy chuẩn đánh giá (EM vs EX trên SQLite) & Export GGUF
│   └── project_structure_and_workflows.md # Chi tiết cấu trúc & quy trình chạy trên Colab vs GPU Server
└── raw_data/                           # Dữ liệu gốc đã tải từ Hugging Face (gretel, spider, bird)
```

### 📚 Danh Mục Tài Liệu Kỹ Thuật Trong [docs/](docs/):
- 🗺️ [**Cấu Trúc & Luồng Thực Thi (project_structure_and_workflows.md)**](docs/project_structure_and_workflows.md): Giải thích chi tiết từng thư mục, quy trình chạy trên Colab (1-Click) vs GPU Server (RunPod/tmux/Wandb).
- 🏗️ [**Kiến Trúc Hệ Thống (architecture.md)**](docs/architecture.md): Phân tích thiết kế Base Model (`Qwen3.5-4B` 4-bit NF4), cơ chế Attention, RoPE scaling và sơ đồ luồng dữ liệu.
- 🧠 [**Cẩm Nang Distillation (distillation_guide.md)**](docs/distillation_guide.md): So sánh Logits KD vs Sequence KD, chắt lọc chuỗi tư duy reasoning kiểu DeepSeek-R1, và mẹo dùng API miễn phí (Groq).
- 🧹 [**Xử Lý & Trộn Dữ Liệu (data_pipeline.md)**](docs/data_pipeline.md): Tiêu chí lọc bỏ basic SQL, chuẩn hóa ChatML messages và chiến lược chống data leakage (Zero-shot DB split).
- ⚡ [**Fine-tuning với Unsloth (finetuning_unsloth.md)**](docs/finetuning_unsloth.md): Giải thích rank, alpha, target modules, kỹ thuật `train_on_responses_only` và bảng tính VRAM cho GPU 8GB - 16GB.
- 📊 [**Đánh Giá & Triển Khai (evaluation_benchmark.md)**](docs/evaluation_benchmark.md): Cách thiết kế Test Harness đo Execution Accuracy trên SQLite và xuất file GGUF cho Ollama.

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh (Quickstart)

### Bước 1: Kích hoạt môi trường ảo
`powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
`

### Bước 2: Tải dữ liệu thô
Dữ liệu đã được hỗ trợ tải tự động qua script:
`powershell
python download_data.py
`

### Bước 3: Lọc câu khó và Hợp nhất dữ liệu
Chạy script tiền xử lý để lọc các câu có JOIN và nhiều điều kiện, xuất ra tập 	rain.jsonl và 	est.jsonl:
`powershell
python preprocess_and_merge.py
`

### Bước 4: Chạy Fine-tuning với Unsloth
Huấn luyện mô hình nền tảng **Qwen/Qwen3.5-4B** bằng QLoRA 4-bit (NF4):
```powershell
python train_unsloth.py
```

---

## 🛠️ Công Nghệ Sử Dụng
- **Mô hình nền tảng**: Qwen/Qwen3.5-4B (Bản 4-bit NF4 Quantization)
- **Fine-tuning Framework**: Unsloth (FastLanguageModel, FastLoraModel) + TRL SFTTrainer
- **Tập dữ liệu**: Gretel Synthetic, Spider, BIRD-Bench
- **Định dạng Chat**: ChatML có Native Reasoning (`<think> ... </think>`)
- **Serving**: GGUF (4-bit Q4_K_M) qua Ollama / vLLM
