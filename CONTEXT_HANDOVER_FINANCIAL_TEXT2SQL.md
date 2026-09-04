# BẢN CHUYỂN GIAO BỐI CẢNH (CONTEXT HANDOVER): TỪ FINETUNE TEXT-TO-SQL SANG FINANCIAL CHATBOT RAG

Tài liệu này lưu trữ toàn bộ tiến trình, kiến thức, mô hình đã huấn luyện và định hướng kiến trúc từ phiên làm việc trước để bàn giao sang Project mới.

---

## 1. THÀNH QUẢ ĐÃ ĐẠT ĐƯỢC (MODEL & PIPELINE)

### 1.1. Mô hình đã Fine-tune & Xuất bản thành công:
* **Base Model**: `Qwen/Qwen3.5-4B`
* **Công nghệ huấn luyện**: Unsloth QLoRA 4-bit (NF4), Loss Masking (`train_on_responses_only`), ChatML hỗ trợ thẻ suy luận Native Reasoning (`<think> ... </think>`).
* **Tập dữ liệu huấn luyện**: 17.000 mẫu nâng cao (chọn lọc từ Gretel Synthetic, Spider Context, BIRD-bench; chỉ giữ câu có JOIN >= 2 bảng, multi-conditions, subqueries, window functions).
* **Kết quả huấn luyện**: Huấn luyện trên NVIDIA RTX 3080 Ti (12GB) với BF16, Loss hội tụ cực đẹp tại mốc **~0.35**.
* **Kho lưu trữ Hugging Face Hub (Public & Sẵn sàng sử dụng)**:
  * Bản LoRA Adapter (85MB): [`https://huggingface.co/giangkh19/qwen3.5-4b-sql`](https://huggingface.co/giangkh19/qwen3.5-4b-sql)
  * Bản Full GGUF Q4_K_M (2.78GB cho Ollama): [`https://huggingface.co/giangkh19/qwen3.5-4b-sql-gguf`](https://huggingface.co/giangkh19/qwen3.5-4b-sql-gguf)
* **Lệnh chạy ngay trên Ollama local**:
  ```bash
  ollama run hf.co/giangkh19/qwen3.5-4b-sql-gguf:Q4_K_M
  ```

---

## 2. DỮ LIỆU TÀI CHÍNH VIỆT NAM ĐÃ KHÁM PHÁ (KHO DỮ LIỆU D:/GURU)

Tại thư mục `D:/GURU/data/` (Báo cáo tài chính của 100 công ty niêm yết qua 10 năm: VJC, ACB, FPT, HPG...):
1. **Dữ liệu thô (BCTC OCR)**: Nằm tại `D:/GURU/data/financial_statements/` (100 mã cổ phiếu).
2. **Dữ liệu đã chuẩn hóa siêu giá trị**:
   * File: `D:/GURU/data/derived/facts_all.csv` (63 MB, 400.512 dòng).
   * Các cột có sẵn: `ticker`, `year`, `report_type` (separate/consolidated), `statement` (balance_sheet, income, cash_flow), `item_code`, `item_label`, `item_label_raw`, `period_key`, `period_label`, `value_vnd` (đã parse sẵn thành FLOAT), `src_table_ids`.
3. **Bộ câu hỏi test thực tế**:
   * File: `D:/GURU/data/questions/questions.jsonl` (1.012 câu hỏi tài chính tiếng Việt).

---

## 3. ĐỊNH HƯỚNG KIẾN TRÚC CHO PROJECT MỚI: FINANCIAL TEXT-TO-SQL RAG

### 3.1. Bối cảnh chuyển đổi:
* Cuộc thi ViFinQA trước đây yêu cầu làm theo hướng **Text-to-Pandas** trên hàng nghìn file CSV rời rạc (rất dễ lỗi cú pháp, tràn giới hạn 800 AST nodes, hay bị ra 0.0 do trượt retrieval).
* **Định hướng mới**: Xây dựng một **Trợ lý AI Tài chính Chuẩn Doanh nghiệp** bằng phương pháp **Text-to-SQL Grounded RAG**.

### 3.2. Thiết kế Database SQLite:
Nạp `facts_all.csv` vào một Database SQLite duy nhất, bổ sung các cột nguồn minh bạch:
```sql
CREATE TABLE financial_facts (
    id INTEGER PRIMARY KEY,
    ticker TEXT,              -- Mã CP: 'VJC', 'ACB', 'FPT'
    year INTEGER,             -- 2018, 2022
    report_type TEXT,         -- 'separate' hoặc 'consolidated'
    statement TEXT,           -- 'income', 'balance_sheet', 'cash_flow'
    item_label TEXT,          -- Tên không dấu: 'lai tien gui'
    item_label_raw TEXT,      -- Tên có dấu: 'Lãi tiền gửi và cho vay'
    period_label TEXT,        -- '31/12/2018'
    value_vnd REAL,           -- Giá trị số thực (VND)
    -- CỘT DẪN NGUỒN MINH BẠCH (GROUNDING):
    source_doc TEXT,          -- 'VJC_financial_statements_2018_separate'
    source_table TEXT,        -- 'table_8 - Báo cáo lưu chuyển tiền tệ'
    page_number INTEGER       -- Trang tham chiếu trong BCTC
);
```

### 3.3. Luồng hoạt động của Chatbot:
1. **User hỏi**: *"Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?"*
2. **Model `Qwen3.5-4B-Text2SQL` suy luận**:
   Sinh câu lệnh SQL:
   ```sql
   SELECT item_label_raw, ABS(value_vnd) / 1000000.0 AS value_m, source_doc, source_table
   FROM financial_facts
   WHERE ticker = 'VJC' AND year = 2018 AND report_type = 'separate' AND item_label LIKE '%lai tien gui%'
   LIMIT 1;
   ```
3. **Database thực thi** (mất < 10ms), trả về: `208253.2` triệu đồng kèm nguồn `table_8`.
4. **Chatbot phản hồi**: Đưa ra con số chính xác 100% kèm trích dẫn nguồn BCTC (Zero Hallucination).

---

## 4. CONVERSATION ID THAM CHIẾU
* **ID Phiên làm việc gốc**: `e9ed203f-d9a7-48f7-9a5d-4ba2fe61d7d5`
* Bạn có thể @mention ID này trong chat Antigravity bất cứ lúc nào.
