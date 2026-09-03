import os
import sys
from datasets import load_dataset

RAW_DATA_DIR = "./raw_data"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

print("=" * 60)
print("BAT DAU TAI 3 TAP DU LIEU TU HUGGING FACE")
print(f"Thu muc luu tru: {os.path.abspath(RAW_DATA_DIR)}")
print("=" * 60)

# 1. GRETEL
print("\n[1/3] Dang tai Gretel Synthetic Text-to-SQL...")
try:
    gretel_ds = load_dataset("gretelai/synthetic_text_to_sql")
    gretel_path = os.path.join(RAW_DATA_DIR, "gretel")
    gretel_ds.save_to_disk(gretel_path)
    print(f"--> [OK] Gretel da luu tai: {gretel_path}")
    if "train" in gretel_ds:
        print(f"    So luong train: {len(gretel_ds['train']):,} dong")
except Exception as e:
    print(f"--> [LOI] Gretel: {e}")

# 2. SPIDER CONTEXT
print("\n[2/3] Dang tai Spider (b-mc2/sql-create-context)...")
try:
    spider_ds = load_dataset("b-mc2/sql-create-context")
    spider_path = os.path.join(RAW_DATA_DIR, "spider_context")
    spider_ds.save_to_disk(spider_path)
    print(f"--> [OK] Spider da luu tai: {spider_path}")
    if "train" in spider_ds:
        print(f"    So luong train: {len(spider_ds['train']):,} dong")
except Exception as e:
    print(f"--> [LOI] Spider: {e}")

# 3. BIRD-BENCH
print("\n[3/3] Dang tai BIRD-Bench...")
bird_names = ["bird-bench/bird-bench", "creamlab/BIRD-SQL", "b-mc2/bird-sql"]
bird_ok = False
for name in bird_names:
    try:
        print(f"    Thu tai {name} ...")
        bird_ds = load_dataset(name)
        bird_path = os.path.join(RAW_DATA_DIR, "bird_bench")
        bird_ds.save_to_disk(bird_path)
        print(f"--> [OK] BIRD-Bench da luu tai: {bird_path}")
        if "train" in bird_ds:
            print(f"    So luong train: {len(bird_ds['train']):,} dong")
        bird_ok = True
        break
    except Exception as e:
        print(f"    Loi {name}: {e}")

if not bird_ok:
    print("--> [CHU Y] Chua tai duoc BIRD-bench, se kiem tra lai.")

print("\n" + "=" * 60)
print("HOAN TAT QUA TRINH TAI!")
print("=" * 60)
