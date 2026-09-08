# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:09:03 2026

@author: priya
"""

import requests
import json
import time
import os

API_KEY  = "PASTE_YOUR_USDA_API_KEY_HERE"
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
OUT_FILE = "scrapers/data/usda_supplements.json"

SUPPLEMENT_QUERIES = [
    "vitamin A", "vitamin B1", "vitamin B2", "vitamin B3",
    "vitamin B6", "vitamin B12", "vitamin C", "vitamin D",
    "vitamin E", "vitamin K", "folate", "biotin",
    "calcium", "magnesium", "zinc", "iron", "selenium",
    "potassium", "sodium", "phosphorus", "iodine", "copper",
    "manganese", "chromium", "molybdenum", "fluoride",
    "omega 3", "omega 6", "fish oil", "flaxseed oil",
    "probiotics", "prebiotics", "fiber supplement",
    "protein supplement", "whey protein", "collagen",
    "melatonin", "coenzyme Q10", "alpha lipoic acid",
    "turmeric", "ashwagandha", "ginseng", "echinacea",
    "elderberry", "garlic supplement", "ginkgo biloba",
    "valerian root", "st johns wort", "milk thistle",
    "green tea extract", "resveratrol", "quercetin",
    "glucosamine", "chondroitin", "MSM supplement",
    "creatine", "L-carnitine", "BCAA", "glutamine"
]

def search_foods(query):
    url = f"{BASE_URL}/foods/search"
    params = {
        "query":    query,
        "dataType": ["Branded", "Foundation", "SR Legacy"],
        "pageSize": 25,
        "api_key":  API_KEY
    }
    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        print(f"  API error: {response.status_code}")
        return []
    return response.json().get("foods", [])

def food_to_markdown(food, query):
    name      = food.get("description", "Unknown")
    brand     = food.get("brandOwner", "")
    category  = food.get("foodCategory", "")
    fdc_id    = food.get("fdcId", "")
    nutrients = food.get("foodNutrients", [])
    ingredients = food.get("ingredients", "")

    lines = []
    lines.append(f"# {name}")
    lines.append(f"> Source: USDA FoodData Central")
    lines.append(f"> FDC ID: {fdc_id}")
    lines.append(f"> Search Query: {query}")
    lines.append("")

    if brand:
        lines.append(f"**Brand:** {brand}")
    if category:
        lines.append(f"**Category:** {category}")
    if food.get("servingSize"):
        lines.append(f"**Serving Size:** {food.get('servingSize')} {food.get('servingSizeUnit','')}")
    lines.append("")

    if ingredients:
        lines.append("## Ingredients")
        lines.append(ingredients)
        lines.append("")

    if nutrients:
        lines.append("## Nutritional Information")
        lines.append("| Nutrient | Amount | Unit |")
        lines.append("| --- | --- | --- |")
        for n in nutrients[:30]:  # top 30 nutrients
            nutrient_name = n.get("nutrientName", "")
            amount        = n.get("value", "")
            unit          = n.get("unitName", "")
            if nutrient_name and amount:
                lines.append(f"| {nutrient_name} | {amount} | {unit} |")
        lines.append("")

    return "\n".join(lines)

def main():
    os.makedirs("scrapers/data", exist_ok=True)

    results  = []
    seen_ids = set()

    # Resume if exists
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            results  = json.load(f)
            seen_ids = {r["fdc_id"] for r in results}
            print(f"▶️  Resuming — {len(results)} already saved")
    except FileNotFoundError:
        print("🆕 Starting fresh...")

    for i, query in enumerate(SUPPLEMENT_QUERIES):
        print(f"[{i+1}/{len(SUPPLEMENT_QUERIES)}] Searching: '{query}'", end=" ")
        foods = search_foods(query)

        added = 0
        for food in foods:
            fdc_id = food.get("fdcId")
            if fdc_id and fdc_id not in seen_ids:
                seen_ids.add(fdc_id)
                md = food_to_markdown(food, query)
                results.append({
                    "name":     food.get("description", ""),
                    "fdc_id":   fdc_id,
                    "source":   "USDA_FDC",
                    "markdown": md,
                    "length":   len(md)
                })
                added += 1

        print(f"✅ +{added} new (total: {len(results)})")
        time.sleep(0.5)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ USDA done! {len(results)} food/supplement items")
    print(f"Saved → {OUT_FILE}")

if __name__ == "__main__":
    main()