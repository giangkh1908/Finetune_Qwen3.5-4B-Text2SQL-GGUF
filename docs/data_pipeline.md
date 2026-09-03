# Quy Trình Xử Lý Dữ Liệu: Lọc, Chuẩn Hóa và Trộn Đa Nguồn (Data Pipeline)

Tài liệu này mô tả chi tiết chiến lược kỹ thuật dữ liệu của dự án: phân tích các tập dữ liệu thô, tiêu chuẩn loại bỏ câu hỏi đơn giản, thiết lập cấu trúc thống nhất (Unified ChatML Schema) và kỹ thuật chia tập Train/Test chống rò rỉ dữ liệu (Anti-Leakage).

---

## 1. Hiện Trạng Dữ Liệu Thô (Raw Data Analysis)

Dữ liệu đã được tải về và lưu trữ tại thư mục `raw_data/`:

```text
raw_data/
├── gretel/                 # 100.000 mẫu train + 5.851 mẫu test
├── spider_context/         # 78.577 mẫu train
└── bird_bench/             # 9.428 mẫu train
```

### 1.1. Tập Gretel Synthetic (`gretelai/synthetic_text_to_sql`)
Phân tích theo cột `sql_complexity`:
* `basic SQL`: 48.466 mẫu (chiếm gần 50% - chỉ có 1 bảng, SELECT đơn giản). $\rightarrow$ **CẦN LOẠI BỎ HẾT!**
* `single join`: 14.932 mẫu (JOIN 2 bảng).
* `multiple_joins`: 2.949 mẫu (JOIN 3 - 5 bảng).
* `subqueries`: 6.719 mẫu (truy vấn lồng nhau).
* `window functions`: 3.596 mẫu (`ROW_NUMBER()`, `RANK()`, `PARTITION BY`).
* `set operations`: 1.050 mẫu (`UNION`, `INTERSECT`, `EXCEPT`).
* `CTEs`: 273 mẫu (`WITH clause AS ...`).
* `aggregation`: 22.015 mẫu (`GROUP BY`, `HAVING`).

### 1.2. Tập Spider Context (`b-mc2/sql-create-context`)
* Tổng số: 78.577 mẫu.
* Có **22.759 mẫu** chứa từ 2 điều kiện lọc trở lên (`AND`/`OR`).
* Có **1.931 mẫu** chứa mệnh đề `JOIN` rõ ràng.

### 1.3. Tập BIRD-Bench (`xu3kev/BIRD-SQL-data-train`)
* Tổng số: 9.428 mẫu.
* **7.212 mẫu có JOIN (chiếm 76.5%)** liên kết giữa nhiều bảng phức tạp.
* **3.694 mẫu** kết hợp nhiều điều kiện logic phức tạp.
* Chứa cột `evidence`: cung cấp giải thích nghiệp vụ thực tế (Domain Knowledge).

---

## 2. Tiêu Chí Lọc Dữ Liệu Phức Tạp (Data Filtering Rules)

Để tránh tình trạng model học vẹt các câu lệnh 1 điều kiện tầm thường, pipeline áp dụng bộ lọc nghiêm ngặt:

```text
       ┌───────────────────────────────────────────────┐
       │             MẪU DỮ LIỆU ĐẦU VÀO               │
       └───────────────────────┬───────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [sql_complexity == 'basic SQL']       [Thỏa mãn >= 1 điều kiện sau]:
            │                          1. Có JOIN (>= 2 bảng)
            │                          2. Có từ 2 điều kiện AND/OR trở lên
            ▼                          3. Có GROUP BY + HAVING
       [LOẠI BỎ]                       4. Có Subquery hoặc CTE (WITH)
                                                  │
                                                  ▼
                                        [CHẤP NHẬN ĐƯA VÀO TRAIN]
```

### Quy tắc làm sạch (Data Cleaning):
1. **Loại bỏ Markdown thừa**: Cắt bỏ các ký tự ```sql hoặc ``` trong cột `sql` để model chỉ sinh câu lệnh thuần túy.
2. **Khử trùng lặp (Deduplication)**: Nếu cùng một câu hỏi và cùng một schema xuất hiện nhiều lần, chỉ giữ lại một câu SQL chuẩn nhất.
3. **Chuẩn hóa khoảng trắng**: Loại bỏ dấu cách thừa, dòng trống trong định nghĩa DDL Schema.

---

## 3. Cấu Trúc Dữ Liệu Thống Nhất (Unified ChatML Schema)

Mỗi mẫu dữ liệu sau khi xử lý sẽ được lưu dưới dạng 1 dòng JSON (`.jsonl`) theo chuẩn OpenAI / ChatML:

```json
{
  "source": "bird_bench",
  "db_id": "video_games",
  "complexity": "multiple_joins",
  "messages": [
    {
      "role": "system",
      "content": "Bạn là chuyên gia cơ sở dữ liệu SQLite. Hãy phân tích Database Schema dưới đây và viết câu lệnh SQL chính xác để trả lời câu hỏi của người dùng. Chỉ trả về câu lệnh SQL, kết thúc bằng dấu chấm phẩy.\n\n### DATABASE SCHEMA:\nCREATE TABLE genre (id INTEGER PRIMARY KEY, genre_name TEXT);\nCREATE TABLE game (id INTEGER PRIMARY KEY, genre_id INTEGER, game_name TEXT, FOREIGN KEY (genre_id) REFERENCES genre(id));\nCREATE TABLE region_sales (game_id INTEGER, region_id INTEGER, num_sales REAL, FOREIGN KEY (game_id) REFERENCES game(id));"
    },
    {
      "role": "user",
      "content": "Liệt kê top 5 thể loại game có doanh số tại khu vực Bắc Mỹ (region_id = 1) lớn hơn 50 triệu bản?\n[Gợi ý]: Doanh số tính bằng triệu bản tương ứng với num_sales > 50;"
    },
    {
      "role": "assistant",
      "content": "SELECT g.genre_name, SUM(rs.num_sales) AS total_na_sales\nFROM genre g\nJOIN game gm ON g.id = gm.genre_id\nJOIN region_sales rs ON gm.id = rs.game_id\nWHERE rs.region_id = 1\nGROUP BY g.id, g.genre_name\nHAVING total_na_sales > 50\nORDER BY total_na_sales DESC\nLIMIT 5;"
    }
  ]
}
```

---

## 4. Tỷ Lệ Trộn Dữ Liệu (Data Mixing & Balancing)

Để đạt được chất lượng tối ưu mà không làm mô hình bị thiên lệch (bias) về một nguồn duy nhất, chúng ta chọn lọc tập train với kích thước lý tưởng **18.000 - 20.000 mẫu**:

| Nguồn Dữ Liệu | Số lượng chọn | Tỷ trọng | Mục đích học tập |
| :--- | :--- | :--- | :--- |
| **Gretel Synthetic** | **8.000** | ~42% | Học hàm nâng cao (Window, CTE, Subquery) và tư duy logic |
| **Spider Context** | **6.000** | ~32% | Học liên kết khóa ngoại và câu truy vấn nhiều điều kiện |
| **BIRD-Bench** | **5.000** | ~26% | Học xử lý schema thực tế của doanh nghiệp và gợi ý nghiệp vụ |
| **TỔNG CỘNG** | **19.000** | **100%** | **Cân bằng hoàn hảo giữa lý thuyết, độ sâu và thực tiễn** |

---

## 5. Chiến Lược Chia Train / Test Chống Rò Rỉ Dữ Liệu (Zero-Shot Split)

### Cạm bẫy rò rỉ dữ liệu (Data Leakage):
Nếu chia ngẫu nhiên 90% train / 10% test theo từng dòng (random sample split), một Database tên là `airline_system` sẽ có 90 câu hỏi trong train và 10 câu hỏi trong test. Model sẽ **học thuộc lòng tên bảng và tên cột** của `airline_system`, dẫn đến việc test điểm rất cao nhưng khi triển khai trên database mới thì hoàn toàn thất bại.

### Giải pháp Zero-shot Database Split:
* Toàn bộ các câu hỏi thuộc cùng một `db_id` (hoặc cùng một Schema) **phải nằm trọn vẹn trong một tập**: hoặc chỉ ở Train, hoặc chỉ ở Test.
* **Tỷ lệ**: 90% số lượng Database dùng để Huấn luyện (Train), 10% số lượng Database dành riêng để Kiểm thử (Test/Eval).
* Bằng cách này, chúng ta đo lường chính xác năng lực **tổng quát hóa trên cơ sở dữ liệu chưa từng thấy** của mô hình.
