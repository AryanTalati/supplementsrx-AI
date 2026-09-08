# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:31:46 2026

@author: priya
"""

import requests
from bs4 import BeautifulSoup
import json

def get_all_monograph_urls():
    url = "https://naturalmedicines.therapeuticresearch.com/IngredientsTherapiesMonographs"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    base = "https://naturalmedicines.therapeuticresearch.com"
    links = []
    
    # Every ingredient link follows the pattern /Data/ProMonographs/Name
    for a in soup.select("a[href*='/Data/ProMonographs/']"):
        name = a.text.strip()
        href = a.get("href")
        if name and href:
            links.append({
                "name": name,
                "url": base + href
            })
    
    # Remove duplicates
    seen = set()
    unique = []
    for item in links:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
    
    with open("data/natmed_urls.json", "w") as f:
        json.dump(unique, f, indent=2)
    
    print(f"✅ Found {len(unique)} ingredients")
    print("Saved to data/natmed_urls.json")
    
    # Preview first 5
    for item in unique[:5]:
        print(f"  - {item['name']} → {item['url']}")

get_all_monograph_urls()