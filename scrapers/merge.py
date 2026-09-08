# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:09:51 2026

@author: priya
"""

import json
import os

FILES = [
    ("scrapers/data/natmed_supplements.json",    "NaturalMedicines"),
    ("scrapers/data/medlineplus_supplements.json","MedlinePlus"),
    ("scrapers/data/usda_supplements.json",       "USDA_FDC"),
    ("scrapers/data/dsld_supplements.json",       "DSLD_NIH"),
]

OUT_JSON = "scrapers/data/all_supplements.json"
OUT_MD   = "scrapers/data/all_supplements.md"

all_data = []

print("Merging all sources...\n")
for filepath, source in FILES:
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        # Only keep entries with real content
        valid = [d for d in data if d.get("markdown") and len(d.get("markdown","")) > 100]
        all_data.extend(valid)
        print(f"✅ {source:20s} → {len(valid):,} records")
    except FileNotFoundError:
        print(f"⏭️  {source:20s} → file not found, skipping")

# Save merged JSON
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

# Save combined markdown
with open(OUT_MD, "w", encoding="utf-8") as f:
    for item in all_data:
        f.write(item.get("markdown", ""))
        f.write("\n\n---\n\n")

print(f"\n{'='*45}")
print(f"  Total records  : {len(all_data):,}")
print(f"  JSON output    : {OUT_JSON}")
print(f"  Markdown output: {OUT_MD}")
print(f"{'='*45}")