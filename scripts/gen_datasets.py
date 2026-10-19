"""Generate benchmark datasets of three sizes + a merged-cell Excel workbook.

Usage:  .venv/Scripts/python scripts/gen_datasets.py [outdir]

Creates:
  small_sales.csv        ~2k rows, 8 cols
  mid_sales.csv          ~60k rows, 10 cols
  big_sales.csv          ~1M rows, 10 cols
  messy_report.xlsx      multi-sheet, merged category headers (ingest stress)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/bench")
out_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)

REGIONS = ["north", "south", "east", "west", "central"]
PRODUCTS = ["widget", "gadget", "gizmo", "doohickey", "thingamajig", "doodad"]
CHANNELS = ["online", "retail", "wholesale"]
SEGMENTS = ["consumer", "business", "enterprise"]


def rows(n: int, cols: int) -> list[list]:
    out = []
    for _ in range(n):
        region = rng.choice(REGIONS)
        product = rng.choice(PRODUCTS)
        channel = rng.choice(CHANNELS)
        segment = rng.choice(SEGMENTS)
        units = rng.randint(1, 60)
        price = round(rng.uniform(2.0, 45.0), 2)
        discount = rng.choice([0, 0, 0, 5, 10, 15])
        month = rng.randint(1, 12)
        revenue = round(units * price * (1 - discount / 100), 2)
        row = [region, product, channel, segment, units, price, discount,
               f"2025-{month:02d}", revenue]
        if cols > 9:
            rating = round(rng.uniform(1.0, 5.0), 1) if rng.random() > 0.05 else ""
            row.append(rating)
        if cols > 10:
            row.append(rng.choice(["A", "B", "C", ""]))
        out.append(row)
    return out


def write_csv(path: Path, n: int, cols: int) -> None:
    header = ["region", "product", "channel", "segment", "units", "unit_price",
              "discount_pct", "month", "revenue"]
    if cols > 9:
        header.append("rating")
    if cols > 10:
        header.append("tier")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows(n, cols):
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {path.name}: {n:,} rows")


write_csv(out_dir / "small_sales.csv", 2_000, 8)
write_csv(out_dir / "mid_sales.csv", 60_000, 10)
write_csv(out_dir / "big_sales.csv", 1_000_000, 10)

# merged-cell Excel stress: categories merged vertically, region headers merged horizontally
try:
    from openpyxl import Workbook
except ImportError:
    print("openpyxl missing — skipped messy_report.xlsx")
    sys.exit(0)

wb = Workbook()
ws = wb.active
ws.title = "summary"
ws["A1"] = "Quarterly Sales Report — FY2025"
ws.merge_cells("A1:E1")
ws["A2"] = "Region"
ws["B2"] = "Product"
ws["C2"] = "Units"
ws["D2"] = "Revenue"
ws["E2"] = "Notes"
row_i = 3
for region in REGIONS[:3]:
    start = row_i
    for product in PRODUCTS[:4]:
        ws.cell(row_i, 2, product)
        ws.cell(row_i, 3, rng.randint(5, 200))
        ws.cell(row_i, 4, round(rng.uniform(100, 9000), 2))
        ws.cell(row_i, 5, "auto-generated")
        row_i += 1
    ws.cell(start, 1, region)
    if row_i - 1 > start:
        ws.merge_cells(start_row=start, start_column=1, end_row=row_i - 1, end_column=1)

ws2 = wb.create_sheet("detail")
ws2.append(["product", "units", "price"])
for product in PRODUCTS:
    ws2.append([product, rng.randint(1, 50), round(rng.uniform(2, 45), 2)])
wb.save(out_dir / "messy_report.xlsx")
print("wrote messy_report.xlsx: merged cells on 'summary' sheet + 'detail' sheet")
