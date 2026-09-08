# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:54:24 2026

@author: priya
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time
import os

# ============================================================
#  ✏️  FILL THESE IN BEFORE RUNNING
# ============================================================
USERNAME = "hdavulcu@asu.edu"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
# ============================================================

LOGIN_URL = "https://login.therapeuticresearch.com/realms/trcSubscribers/protocol/openid-connect/auth?client_id=SitecoreClient&response_type=code&scope=openid&redirect_uri=https%3a%2f%2fnaturalmedicines.therapeuticresearch.com%2fapi%2fsitecore%2fOidcCallback%2fCallback&state=L0luZ3JlZGllbnRzVGhlcmFwaWVzTW9ub2dyYXBocw2&nonce=04d8b262e9d94674b5586ca8d4d0b2e3"
URLS_FILE = f"C:/Users/priya/Documents/Study/ASU/Sem 2/Semantic Web Mining/Project_Work/scrapers/data/natmed_urls.json"
JSON_OUT  = f"C:/Users/priya/Documents/Study/ASU/Sem 2/Semantic Web Mining/Project_Work/scrapers/data/natmed_supplements.json"
MD_DIR    = f"C:/Users/priya/Documents/Study/ASU/Sem 2/Semantic Web Mining/Project_Work/scrapers/data/markdown"
MD_ALL    = f"C:/Users/priya/Documents/Study/ASU/Sem 2/Semantic Web Mining/Project_Work/scrapers/data/markdown/all_supplements.md"


# ============================================================
#  BROWSER SETUP
# ============================================================
def create_driver():
    options = webdriver.ChromeOptions()
    # Remove the # below to hide the browser window
    # options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver


# ============================================================
#  LOGIN
# ============================================================
def login(driver):
    print("🔐 Logging in...")
    driver.get(LOGIN_URL)
    wait = WebDriverWait(driver, 20)

    try:
        # Wait for username field to be fully clickable (not just present)
        username_field = wait.until(EC.element_to_be_clickable((By.ID, "username")))
        username_field.clear()
        username_field.send_keys(USERNAME)
        time.sleep(1)

        # Wait for password field to be clickable
        password_field = wait.until(EC.element_to_be_clickable((By.ID, "password")))
        password_field.clear()
        password_field.send_keys(PASSWORD)
        time.sleep(1)

        # Wait for login button to be clickable
        login_btn = wait.until(EC.element_to_be_clickable((By.ID, "kc-login")))
        login_btn.click()
        time.sleep(5)

        if "login" in driver.current_url.lower():
            print("❌ Login failed — check your USERNAME and PASSWORD")
            return False

        print("✅ Logged in successfully!\n")
        return True

    except Exception as e:
        print(f"❌ Login error: {e}")
        return False


# ============================================================
#  SCRAPE ONE PAGE
# ============================================================
def scrape_page(driver, name, url):
    try:
        driver.get(url)
        
        # Wait longer for JS to fully render the page
        time.sleep(4)

        # Re-login if session expired
        if "login" in driver.current_url.lower():
            print("  ⚠️  Session expired — re-logging in...")
            if not login(driver):
                return "LOGIN_FAILED"
            driver.get(url)
            time.sleep(5)

        # Scroll down slowly to trigger any lazy-loaded content
        for scroll in range(5):
            driver.execute_script(f"window.scrollTo(0, {scroll * 500});")
            time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Remove all noise — nav, footer, scripts, ads
        for tag in soup.find_all(["nav", "header", "footer", "script", 
                                   "style", "noscript", "iframe", "button"]):
            tag.decompose()

        # ── Try to find the main content area ──
        main = (
            soup.find("main") or
            soup.find("div", id="main-content") or
            soup.find("div", id="content") or
            soup.find("div", id="main") or
            soup.find("article") or
            soup.find("div", class_=lambda c: c and any(
                x in " ".join(c).lower() 
                for x in ["content", "monograph", "detail", "page-body", "main"]
            ) if c else False) or
            soup.find("body")
        )

        # ── Extract all headings and their content ──
        sections = {}
        current_heading = "General"
        current_content = []

        if main:
            for tag in main.find_all(["h1","h2","h3","h4","p","ul","ol","table","div"]):
                # Skip deeply nested divs (only top level)
                if tag.name == "div" and tag.find_parent(["div"]) != main:
                    continue

                if tag.name in ["h1","h2","h3","h4"]:
                    # Save previous section
                    if current_content:
                        text = " ".join(current_content).strip()
                        if text:
                            sections[current_heading] = text
                    current_heading = tag.get_text(strip=True)
                    current_content = []

                elif tag.name in ["p","ul","ol"]:
                    text = tag.get_text(separator="\n", strip=True)
                    if text:
                        current_content.append(text)

                elif tag.name == "table":
                    # Convert table to markdown
                    rows = tag.find_all("tr")
                    table_lines = []
                    for i, row in enumerate(rows):
                        cells = row.find_all(["th","td"])
                        line = " | ".join(c.get_text(strip=True) for c in cells)
                        table_lines.append(f"| {line} |")
                        if i == 0:
                            separator = " | ".join(["---"] * len(cells))
                            table_lines.append(f"| {separator} |")
                    if table_lines:
                        current_content.append("\n".join(table_lines))

            # Save last section
            if current_content:
                text = " ".join(current_content).strip()
                if text:
                    sections[current_heading] = text

        # ── Fallback: if no sections found, grab all text ──
        if not sections and main:
            full_text = main.get_text(separator="\n", strip=True)
            sections["Full Content"] = full_text

        # ── Build data dict ──
        data = {
            "name":     name,
            "url":      url,
            "source":   "NaturalMedicines",
            "sections": sections,
            "llm_text": ""
        }

        data["llm_text"] = build_llm_text(name, url, sections)
        return data

    except Exception as e:
        print(f"  ❌ Error scraping {name}: {e}")
        return None




# ============================================================
#  BUILD LLM-FRIENDLY TEXT
# ============================================================
def build_llm_text(name, url, sections):
    lines = []
    lines.append(f"# {name}")
    lines.append(f"> Source: {url}")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not sections:
        lines.append("*No content extracted — page may require additional login.*")
        return "\n".join(lines)

    for heading, content in sections.items():
        if heading and content.strip():
            lines.append(f"## {heading}")
            lines.append(content.strip())
            lines.append("")

    return "\n".join(lines)


# ============================================================
#  SAVE MARKDOWN FILES
# ============================================================
def save_markdown(results):
    os.makedirs(MD_DIR, exist_ok=True)
    print(f"\n📝 Saving markdown files...")

    for item in results:
        filename = item.get("name", "unknown")
        filename = "".join(c if c.isalnum() or c in " -_" else "_" for c in filename)
        filename = filename.strip().replace(" ", "_") + ".md"
        filepath = os.path.join(MD_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(item.get("llm_text", ""))

    with open(MD_ALL, "w", encoding="utf-8") as f:
        for item in results:
            f.write(item.get("llm_text", ""))
            f.write("\n\n---\n\n")

    print(f"✅ Individual .md files → {MD_DIR}/  ({len(results)} files)")
    print(f"✅ Combined file        → {MD_ALL}")

# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 50)
    print("  NaturalMedicines Complete Scraper")
    print("=" * 50)

    # Load URL list
    with open(URLS_FILE) as f:
        url_list = json.load(f)
    print(f"\n📋 Total ingredients in URL list: {len(url_list)}")

    # Load existing progress
    results   = []
    done_urls = set()
    try:
        with open(JSON_OUT, encoding="utf-8") as f:
            existing = json.load(f)
        results   = [r for r in existing if r.get("llm_text") or r.get("full_text") or r.get("overview")]
        done_urls = {r["url"] for r in results}
        url_list  = [u for u in url_list if u["url"] not in done_urls]
        print(f"▶️  Resuming  — {len(results)} already done, {len(url_list)} remaining\n")
    except FileNotFoundError:
        print("🆕 Starting fresh...\n")

    if not url_list:
        print("✅ All ingredients already scraped! Saving markdown...")
        save_markdown(results)
        return

    # Start browser and login
    driver = create_driver()
    if not login(driver):
        driver.quit()
        return

    # Scrape all pages
    try:
        for i, item in enumerate(url_list):
            print(f"[{i+1}/{len(url_list)}] {item['name']}")
            result = scrape_page(driver, item["name"], item["url"])

            if result == "LOGIN_FAILED":
                print("⛔ Cannot recover. Saving progress and stopping.")
                break

            if result:
                results.append(result)

            # Save JSON progress every 20 items
            if (i + 1) % 20 == 0:
                with open(JSON_OUT, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  💾 Progress saved ({len(results)} total)\n")

            time.sleep(1.5)

    finally:
        # Always save on exit even if it crashes
        print("\n💾 Saving final JSON...")
        with open(JSON_OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save markdown
        save_markdown(results)

        print(f"\n{'='*50}")
        print(f"  ✅ COMPLETE!")
        print(f"  Total scraped : {len(results)}")
        print(f"  JSON output   : {JSON_OUT}")
        print(f"  Markdown dir  : {MD_DIR}/")
        print(f"  Combined .md  : {MD_ALL}")
        print(f"{'='*50}")

        driver.quit()


if __name__ == "__main__":
    main()