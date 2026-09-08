# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:09:03 2026

@author: priya
"""
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup, Comment
import json, time, os, re

# ============================================================
USERNAME = "hdavulcu@asu.edu"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
# ============================================================

LOGIN_URL = "https://naturalmedicines.therapeuticresearch.com/api/sitecore/account/SingleSignOn/?url=%2fHome%2fND"
URLS_FILE = "scrapers/data/natmed_urls.json"
JSON_OUT  = "scrapers/data/natmed_supplements.json"
MD_DIR    = "scrapers/data/markdown"
MD_ALL    = "scrapers/data/all_supplements.md"

# ── STEP 1: Browser ──────────────────────────────────────────
def create_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")   # uncomment to hide browser
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

# ── STEP 2: Login ─────────────────────────────────────────────
def login(driver):
    print("🔐 Logging in...")
    driver.get(LOGIN_URL)
    wait = WebDriverWait(driver, 20)
    try:
        wait.until(EC.element_to_be_clickable((By.ID, "username"))).send_keys(USERNAME)
        time.sleep(0.5)
        wait.until(EC.element_to_be_clickable((By.ID, "password"))).send_keys(PASSWORD)
        time.sleep(0.5)
        wait.until(EC.element_to_be_clickable((By.ID, "kc-login"))).click()
        time.sleep(5)
        if "login" in driver.current_url.lower():
            print("❌ Login failed — check USERNAME and PASSWORD")
            return False
        print("✅ Logged in!\n")
        return True
    except Exception as e:
        print(f"❌ Login error: {e}")
        return False

# ── STEP 3: Clean HTML → Markdown ────────────────────────────
def html_to_markdown(soup, name, url):
    """
    Convert full page HTML to clean markdown.
    Dumps ALL visible text, structured by headings.
    """
    # Remove all noise tags completely
    for tag in soup.find_all([
        "script", "style", "noscript", "iframe",
        "nav", "header", "footer", "button",
        "form", "input", "select", "textarea",
        "svg", "img", "figure", "video", "audio"
    ]):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Find main content — try multiple selectors
    main = (
        soup.find("div", id="main-content") or
        soup.find("div", id="content")      or
        soup.find("div", id="main")         or
        soup.find("main")                   or
        soup.find("article")               or
        soup.find("div", class_=re.compile(r"(content|monograph|detail|page-body)", re.I)) or
        soup.find("body")
    )

    if not main:
        return f"# {name}\n\n> Source: {url}\n\n*No content found.*\n"

    lines = []
    lines.append(f"# {name}")
    lines.append(f"> Source: {url}")
    lines.append("")

    def process_node(node):
        """Recursively walk the DOM and convert to markdown"""
        if not node:
            return

        # Skip hidden elements
        if hasattr(node, "get"):
            style = node.get("style", "")
            if "display:none" in style.replace(" ", "") or \
               "visibility:hidden" in style.replace(" ", ""):
                return

        tag = node.name if hasattr(node, "name") else None

        # Text node — just add the text
        if tag is None:
            text = str(node).strip()
            if text:
                lines.append(text)
            return

        # Headings → markdown headings
        if tag == "h1":
            text = node.get_text(strip=True)
            if text:
                lines.append(f"\n# {text}")
        elif tag == "h2":
            text = node.get_text(strip=True)
            if text:
                lines.append(f"\n## {text}")
        elif tag == "h3":
            text = node.get_text(strip=True)
            if text:
                lines.append(f"\n### {text}")
        elif tag == "h4":
            text = node.get_text(strip=True)
            if text:
                lines.append(f"\n#### {text}")

        # Paragraphs
        elif tag == "p":
            text = node.get_text(separator=" ", strip=True)
            if text:
                lines.append(f"\n{text}\n")

        # Unordered list
        elif tag == "ul":
            lines.append("")
            for li in node.find_all("li", recursive=False):
                text = li.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"- {text}")
            lines.append("")

        # Ordered list
        elif tag == "ol":
            lines.append("")
            for i, li in enumerate(node.find_all("li", recursive=False), 1):
                text = li.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"{i}. {text}")
            lines.append("")

        # Tables → markdown table
        elif tag == "table":
            lines.append("")
            rows = node.find_all("tr")
            for i, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue
                cell_texts = [c.get_text(separator=" ", strip=True) for c in cells]
                lines.append("| " + " | ".join(cell_texts) + " |")
                if i == 0:
                    lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
            lines.append("")

        # Bold / strong
        elif tag in ["strong", "b"]:
            text = node.get_text(strip=True)
            if text:
                lines.append(f"**{text}**")

        # Italic / em
        elif tag in ["em", "i"]:
            text = node.get_text(strip=True)
            if text:
                lines.append(f"*{text}*")

        # Divs, sections, spans — recurse into children
        elif tag in ["div", "section", "article", "main",
                     "span", "td", "th", "li"]:
            for child in node.children:
                process_node(child)

        # Line break
        elif tag == "br":
            lines.append("")

        # Horizontal rule
        elif tag == "hr":
            lines.append("\n---\n")

        # Anything else — just get the text
        else:
            text = node.get_text(separator=" ", strip=True)
            if text:
                lines.append(text)

    # Process all top-level children of main content
    for child in main.children:
        process_node(child)

    # Clean up: remove excessive blank lines
    result = "\n".join(lines)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")

    return result.strip()


# ── STEP 4: Scrape One Page ───────────────────────────────────
def scrape_page(driver, name, url):
    try:
        driver.get(url)
        time.sleep(3)

        # Re-login if redirected
        if "login" in driver.current_url.lower():
            print("  ⚠️  Session expired — re-logging in...")
            if not login(driver):
                return "LOGIN_FAILED"
            driver.get(url)
            time.sleep(4)

        # Scroll to load any lazy content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Check page actually loaded content (not blank/error)
        page_text = soup.get_text(strip=True)
        if len(page_text) < 200:
            print(f"  ⚠️  Page seems empty ({len(page_text)} chars) — skipping")
            return None

        md = html_to_markdown(soup, name, url)

        return {
            "name":     name,
            "url":      url,
            "source":   "NaturalMedicines",
            "markdown": md,
            "length":   len(md)
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


# ── STEP 5: Save Markdown Files ───────────────────────────────
def save_markdown(results):
    os.makedirs(MD_DIR, exist_ok=True)
    print(f"\n📝 Saving {len(results)} markdown files...")

    saved = 0
    for item in results:
        if not item.get("markdown"):
            continue
        fname = item.get("name", "unknown")
        fname = "".join(c if c.isalnum() or c in " -_" else "_" for c in fname)
        fname = fname.strip().replace(" ", "_") + ".md"
        fpath = os.path.join(MD_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(item["markdown"])
        saved += 1

    # Combined file
    with open(MD_ALL, "w", encoding="utf-8") as f:
        for item in results:
            if item.get("markdown"):
                f.write(item["markdown"])
                f.write("\n\n---\n\n")

    print(f"✅ {saved} individual files → {MD_DIR}/")
    print(f"✅ Combined file          → {MD_ALL}")


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   NaturalMedicines → Markdown Scraper")
    print("=" * 55)

    os.makedirs("scrapers/data/markdown", exist_ok=True)

    # Load URLs
    with open(URLS_FILE, encoding="utf-8") as f:
        url_list = json.load(f)
    print(f"\n📋 Total URLs: {len(url_list)}")

    # Resume progress
    results   = []
    done_urls = set()
    try:
        with open(JSON_OUT, encoding="utf-8") as f:
            existing = json.load(f)
        # Only keep entries with actual markdown content
        results   = [r for r in existing if r.get("markdown") and r["length"] > 300]
        done_urls = {r["url"] for r in results}
        url_list  = [u for u in url_list if u["url"] not in done_urls]
        print(f"▶️  Resuming — {len(results)} done, {len(url_list)} remaining\n")
    except FileNotFoundError:
        print("🆕 Starting fresh...\n")

    if not url_list:
        print("✅ All done! Saving markdown...")
        save_markdown(results)
        return

    driver = create_driver()
    if not login(driver):
        driver.quit()
        return

    # ── Test on Ashwagandha first ──
    print("🧪 Testing on Ashwagandha first to verify output...\n")
    test = scrape_page(
        driver,
        "Ashwagandha",
        "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Ashwagandha"
    )
    if test and test.get("markdown") and test["length"] > 300:
        print(f"✅ Test passed! Got {test['length']} chars of markdown")
        print("\n── PREVIEW (first 500 chars) ──")
        print(test["markdown"][:500])
        print("── END PREVIEW ──\n")
        # Save test file so you can check it
        with open("scrapers/data/ashwagandha_test.md", "w", encoding="utf-8") as f:
            f.write(test["markdown"])
        print("📄 Full test saved → scrapers/data/ashwagandha_test.md\n")
    else:
        print("❌ Test failed — page content not loading correctly")
        print(f"   Got: {test}")
        driver.quit()
        return

    input("✋ Check ashwagandha_test.md — if it looks good, press Enter to scrape all 1437...\n")

    try:
        for i, item in enumerate(url_list):
            print(f"[{i+1}/{len(url_list)}] {item['name']}", end=" ")
            result = scrape_page(driver, item["name"], item["url"])

            if result == "LOGIN_FAILED":
                print("\n⛔ Stopping — saving progress.")
                break

            if result and result.get("markdown") and result["length"] > 300:
                results.append(result)
                print(f"✅ {result['length']} chars")
            else:
                print("⚠️  empty/short")

            # Save every 25 items
            if (i + 1) % 25 == 0:
                with open(JSON_OUT, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                save_markdown(results)
                print(f"\n💾 Progress saved — {len(results)} total\n")

            time.sleep(1.5)

    finally:
        print(f"\n💾 Saving final output...")
        with open(JSON_OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        save_markdown(results)

        print(f"\n{'=' * 55}")
        print(f"  ✅ COMPLETE!")
        print(f"  Scraped        : {len(results)} ingredients")
        print(f"  JSON output    : {JSON_OUT}")
        print(f"  Markdown files : {MD_DIR}/")
        print(f"  Combined .md   : {MD_ALL}")
        print(f"{'=' * 55}")
        driver.quit()

if __name__ == "__main__":
    main()