# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:09:32 2026

@author: priya
"""

import requests
import json
import time
import os

API_BASE = "https://api.ods.od.nih.gov/dsld/v9"
OUT_FILE = "scrapers/data/dsld_supplements.json"

def get_products(offset=0, limit=50):
    url = f"{API_BASE}/browse-products"
    params = {
        "offset": offset,
        "limit":  limit,
        "lang":   "en"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers, timeout=15)
    if response.status_code != 200:
        print(f"  API error: {response.status_code} — {response.text[:100]}")
        return []
    data = response.json()
    return data.get("hits", []) or data.get("data", []) or []

def product_to_markdown(p):
    name     = p.get("productName", p.get("name", "Unknown"))
    brand    = p.get("brandName",   p.get("brand", ""))
    form     = p.get("form",        "")
    serving  = p.get("servingSize", "")
    dsld_id  = p.get("dsldId",      p.get("id", ""))
    targets  = p.get("targetGroups", [])
    ingredients = p.get("ingredients", [])

    lines = []
    lines.append(f"# {name}")
    lines.append(f"> Source: NIH DSLD (Dietary Supplement Label Database)")
    lines.append(f"> DSLD ID: {dsld_id}")
    lines.append("")

    if brand:
        lines.append(f"**Brand:** {brand}")
    if form:
        lines.append(f"**Form:** {form}")
    if serving:
        lines.append(f"**Serving Size:** {serving}")
    if targets:
        t = targets if isinstance(targets, str) else ", ".join(targets)
        lines.append(f"**Target Groups:** {t}")
    lines.append("")

    if ingredients:
        lines.append("## Ingredients")
        if isinstance(ingredients, list):
            for ing in ingredients:
                if isinstance(ing, dict):
                    ing_name   = ing.get("name", "")
                    ing_amount = ing.get("quantity", ing.get("amount", ""))
                    ing_unit   = ing.get("unit", "")
                    if ing_name:
                        lines.append(f"- {ing_name}: {ing_amount} {ing_unit}".strip())
                else:
                    lines.append(f"- {ing}")
        else:
            lines.append(str(ingredients))
        lines.append("")

    return "\n".join(lines)

def main():
    os.makedirs("scrapers/data", exist_ok=True)

    results = []
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
            print(f"▶️  Resuming — {len(results)} already saved")
    except FileNotFoundError:
        print("🆕 Starting fresh...")

    MAX_PRODUCTS = 2000
    offset       = len(results)
    empty_count  = 0

    while offset < MAX_PRODUCTS:
        print(f"Fetching products {offset}–{offset+50}...", end=" ")
        products = get_products(offset=offset, limit=50)

        if not products:
            empty_count += 1
            print("empty batch")
            if empty_count >= 3:
                print("3 empty batches in a row — stopping")
                break
            time.sleep(2)
            continue

        empty_count = 0
        for p in products:
            md = product_to_markdown(p)
            results.append({
                "name":     p.get("productName", p.get("name", "")),
                "dsld_id":  p.get("dsldId", p.get("id", "")),
                "source":   "DSLD_NIH",
                "markdown": md,
                "length":   len(md)
            })

        print(f"✅ +{len(products)} (total: {len(results)})")

        # Save every 200
        if len(results) % 200 == 0:
            with open(OUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Progress saved\n")

        offset     += 50
        time.sleep(0.5)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ DSLD done! {len(results)} supplement products")
    print(f"Saved → {OUT_FILE}")

if __name__ == "__main__":
    main()