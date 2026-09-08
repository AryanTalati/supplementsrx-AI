# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:07:45 2026

@author: priya
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os

OUT_FILE = "scrapers/data/medlineplus_supplements.json"
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

BASE = "https://medlineplus.gov"

# All drug letter pages
DRUG_LETTERS = [
    "drug_Aa","drug_Ba","drug_Ca","drug_Da","drug_Ea","drug_Fa",
    "drug_Ga","drug_Ha","drug_Ia","drug_Ja","drug_Ka","drug_La",
    "drug_Ma","drug_Na","drug_Oa","drug_Pa","drug_Qa","drug_Ra",
    "drug_Sa","drug_Ta","drug_Ua","drug_Va","drug_Wa","drug_Xa",
    "drug_Ya","drug_Za","drug_00"
]

# ── GET ALL DRUG LINKS ────────────────────────────────────────
def get_drug_links():
    print("\n📦 Collecting drug links from MedlinePlus...")
    all_links = []
    seen      = set()

    for letter in DRUG_LETTERS:
        url = f"{BASE}/druginfo/{letter}.html"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                name = a.get_text(strip=True)
                if not href or not name:
                    continue
                # Drug pages follow pattern ./meds/aXXXXXX.html
                if "/meds/" not in href and "meds/" not in href:
                    continue
                # Build full URL
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("/"):
                    full_url = BASE + href
                else:
                    full_url = BASE + "/druginfo/" + href.lstrip("./")

                if full_url not in seen:
                    seen.add(full_url)
                    all_links.append({"name": name, "url": full_url, "type": "drug"})

            print(f"  {letter}: {len(all_links)} total so far")
            time.sleep(0.3)

        except Exception as e:
            print(f"  ❌ Error on {letter}: {e}")

    print(f"✅ Found {len(all_links)} drugs total")
    return all_links

# ── GET ALL SUPPLEMENT LINKS ──────────────────────────────────
def get_supplement_links():
    print("\n🌿 Collecting supplement links from MedlinePlus...")
    url  = f"{BASE}/druginfo/herb_All.html"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    seen  = set()

    valid_domains = [
        "nccih.nih.gov/health/",
        "ods.od.nih.gov/factsheets/",
        "cancer.gov/about-cancer/treatment/cam/"
    ]

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        name = a.get_text(strip=True)

        if not href or not name:
            continue

        # Must be one of the 3 NIH domains
        if not any(d in href for d in valid_domains):
            continue

        # ── KEY FIX: always use href as-is since it's already a full URL ──
        if not href.startswith("http"):
            href = "https://" + href

        if href not in seen:
            seen.add(href)
            links.append({"name": name, "url": href, "type": "supplement"})

    print(f"✅ Found {len(links)} supplements total")
    return links

# ── SCRAPE ONE PAGE ───────────────────────────────────────────
def scrape_page(name, url, page_type):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup.find_all(["nav","header","footer","script",
                                   "style","noscript","aside","form",
                                   "button","iframe","figure"]):
            tag.decompose()

        # Find main content based on site
        if "medlineplus.gov" in url:
            main = (soup.find(id="mplus-content") or
                    soup.find(id="main-content")  or
                    soup.find("article")          or
                    soup.find("main"))
        elif "nccih.nih.gov" in url:
            main = (soup.find(id="content-area")   or
                    soup.find("main")               or
                    soup.find("article"))
        elif "ods.od.nih.gov" in url:
            main = (soup.find(id="content")  or
                    soup.find("main")        or
                    soup.find("article"))
        elif "cancer.gov" in url:
            main = (soup.find(id="cgvBody") or
                    soup.find("main")       or
                    soup.find("article"))
        else:
            main = soup.find("main") or soup.find("body")

        if not main:
            main = soup.find("body")

        # Build markdown
        lines = []
        lines.append(f"# {name}")
        lines.append(f"> Source: {url}")
        lines.append(f"> Type: {page_type}")
        lines.append("")

        for tag in main.find_all(
            ["h1","h2","h3","h4","p","ul","ol","table","dl"],
            recursive=True
        ):
            if tag.name in ["ul","ol"] and tag.find_parent(["ul","ol"]):
                continue
            if tag.name == "dl" and tag.find_parent("dl"):
                continue
            if tag.find_parent(["nav","header","footer","aside"]):
                continue

            if tag.name in ["h1","h2","h3","h4"]:
                text  = tag.get_text(strip=True)
                level = int(tag.name[1])
                if text:
                    lines.append(f"\n{'#' * level} {text}")

            elif tag.name == "p":
                text = tag.get_text(separator=" ", strip=True)
                if text and len(text) > 10:
                    lines.append(f"\n{text}")

            elif tag.name in ["ul","ol"]:
                lines.append("")
                for li in tag.find_all("li", recursive=False):
                    text = li.get_text(separator=" ", strip=True)
                    if text:
                        prefix = "- " if tag.name == "ul" else "1. "
                        lines.append(f"{prefix}{text}")
                lines.append("")

            elif tag.name == "dl":
                lines.append("")
                for child in tag.children:
                    if not hasattr(child, "name"):
                        continue
                    if child.name == "dt":
                        lines.append(f"\n**{child.get_text(strip=True)}**")
                    elif child.name == "dd":
                        lines.append(child.get_text(separator=" ", strip=True))
                lines.append("")

            elif tag.name == "table":
                if tag.find_parent("table"):
                    continue
                rows = tag.find_all("tr")
                for i, row in enumerate(rows):
                    cells = row.find_all(["th","td"])
                    if not cells:
                        continue
                    cell_texts = [c.get_text(strip=True) for c in cells]
                    lines.append("| " + " | ".join(cell_texts) + " |")
                    if i == 0:
                        lines.append("| " + " | ".join(["---"]*len(cells)) + " |")
                lines.append("")

        markdown = "\n".join(lines)
        while "\n\n\n" in markdown:
            markdown = markdown.replace("\n\n\n", "\n\n")

        return {
            "name":     name,
            "url":      url,
            "type":     page_type,
            "source":   "MedlinePlus",
            "markdown": markdown.strip(),
            "length":   len(markdown)
        }

    except Exception as e:
        print(f"  ❌ {e}")
        return None

# ── MAIN ─────────────────────────────────────────────────────
def main():
    os.makedirs("scrapers/data", exist_ok=True)

    # Collect all links
    drug_links       = get_drug_links()
    supplement_links = get_supplement_links()
    all_links        = drug_links + supplement_links

    print(f"\n📋 Total to scrape: {len(all_links)}")
    print(f"   Drugs      : {len(drug_links)}")
    print(f"   Supplements: {len(supplement_links)}")

    # Resume support
    results   = []
    done_urls = set()
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            existing  = json.load(f)
            results   = [r for r in existing if r.get("length", 0) > 200]
            done_urls = {r["url"] for r in results}
            all_links = [l for l in all_links if l["url"] not in done_urls]
            print(f"▶️  Resuming — {len(results)} done, {len(all_links)} remaining")
    except FileNotFoundError:
        print("🆕 Starting fresh...")

    # Test on first item
    test_item = all_links[0]
    print(f"\n🧪 Testing: {test_item['name']} ({test_item['type']})")
    test = scrape_page(test_item["name"], test_item["url"], test_item["type"])
    if test and test["length"] > 200:
        print(f"✅ Test passed! {test['length']} chars")
        print("\n── PREVIEW ──")
        print(test["markdown"][:500])
        print("── END PREVIEW ──\n")
    else:
        print(f"❌ Test failed")
        return

    # Scrape all
    failed = []
    for i, item in enumerate(all_links):
        print(f"[{i+1}/{len(all_links)}] {item['name'][:45]}", end=" ")
        result = scrape_page(item["name"], item["url"], item["type"])

        if result and result["length"] > 100:
            results.append(result)
            print(f"✅ {result['length']} chars")
        else:
            failed.append(item)
            print("⚠️  empty")

        if (i + 1) % 50 == 0:
            with open(OUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved {len(results)} total\n")

        time.sleep(0.8)

    # Final save
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if failed:
        with open("scrapers/data/medlineplus_failed.json", "w") as f:
            json.dump(failed, f, indent=2)
        print(f"⚠️  {len(failed)} failed → scrapers/data/medlineplus_failed.json")

    drugs_count = sum(1 for r in results if r.get("type") == "drug")
    supps_count = sum(1 for r in results if r.get("type") == "supplement")

    print(f"\n{'='*50}")
    print(f"  ✅ DONE!")
    print(f"  Drugs scraped      : {drugs_count}")
    print(f"  Supplements scraped: {supps_count}")
    print(f"  Total              : {len(results)}")
    print(f"  Saved → {OUT_FILE}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()