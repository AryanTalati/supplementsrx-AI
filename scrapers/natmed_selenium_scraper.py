# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:46:49 2026

@author: priya
"""
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

# ============================================================
#  PUT YOUR LOGIN CREDENTIALS HERE
# ============================================================
USERNAME = " hdavulcu@asu.edu"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

LOGIN_URL  = "https://naturalmedicines.therapeuticresearch.com/api/sitecore/account/SingleSignOn/?url=%2fHome%2fND"
BASE_URL   = "https://naturalmedicines.therapeuticresearch.com"

def create_driver():
    """Create a Chrome browser that looks like a real user"""
    options = webdriver.ChromeOptions()
    # Comment out the next line if you want to SEE the browser while it runs
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver

def login(driver):
    """Log into NaturalMedicines automatically"""
    print("Logging in...")
    driver.get(LOGIN_URL)
    wait = WebDriverWait(driver, 15)

    # Wait for username field and fill it
    username_field = wait.until(EC.presence_of_element_located((By.ID, "username")))
    username_field.clear()
    username_field.send_keys(USERNAME)

    # Fill password
    password_field = driver.find_element(By.ID, "password")
    password_field.clear()
    password_field.send_keys(PASSWORD)

    # Click login button
    login_btn = driver.find_element(By.ID, "kc-login")
    login_btn.click()

    # Wait until we're past the login page
    time.sleep(4)

    # Verify login worked
    if "login" in driver.current_url.lower():
        print("❌ Login failed — check your USERNAME and PASSWORD")
        driver.quit()
        return False

    print("✅ Login successful!")
    return True

def scrape_page(driver, name, url):
    """Scrape one ingredient page using Selenium + BeautifulSoup"""
    try:
        driver.get(url)
        time.sleep(2)  # Wait for page to fully load

        # Check if redirected to login again (session issue)
        if "login" in driver.current_url.lower():
            print("  ⚠️  Session expired — re-logging in...")
            if not login(driver):
                return "LOGIN_FAILED"
            driver.get(url)
            time.sleep(2)

        # Get fully rendered page HTML
        soup = BeautifulSoup(driver.page_source, "html.parser")

        data = {
            "name":                name,
            "url":                 url,
            "source":              "NaturalMedicines",
            "overview":            "",
            "uses_effectiveness":  [],
            "safety_rating":       "",
            "safety_details":      "",
            "interactions":        [],
            "interactions_text":   "",
            "dosage":              "",
            "adverse_effects":     [],
            "pregnancy_safety":    "",
            "mechanism_of_action": "",
            "brand_names":         [],
            "also_known_as":       []
        }

        # ---- Helper functions ----
        def get_text(id_name):
            el = soup.find(id=id_name)
            return el.get_text(separator=" ", strip=True) if el else ""

        def get_list(id_name):
            el = soup.find(id=id_name)
            if not el:
                return []
            return [li.get_text(strip=True) for li in el.find_all("li")]

        # ---- Try getting sections by ID first ----
        data["overview"]            = get_text("overview")
        data["uses_effectiveness"]  = get_list("uses")
        data["safety_details"]      = get_text("safety")
        data["interactions_text"]   = get_text("interactions")
        data["interactions"]        = get_list("interactions")
        data["dosage"]              = get_text("dosing")
        data["adverse_effects"]     = get_list("adverse-effects")
        data["pregnancy_safety"]    = get_text("pregnancy")
        data["mechanism_of_action"] = get_text("mechanism-of-action")

        # ---- Safety rating badge ----
        for cls in ["safety-rating", "safety-level", "rating", "effectiveness-rating"]:
            badge = soup.find(class_=lambda c: c and cls in c.lower() if c else False)
            if badge:
                data["safety_rating"] = badge.get_text(strip=True)
                break

        # ---- If IDs didn't work, try by heading text ----
        if not data["overview"]:
            for heading in soup.find_all(["h2", "h3", "h4"]):
                heading_text = heading.get_text(strip=True).lower()
                # Get the next sibling div/p as the content
                content_block = heading.find_next_sibling(["div", "p", "ul"])
                if not content_block:
                    parent = heading.parent
                    content_block = parent

                content = content_block.get_text(separator=" ", strip=True) if content_block else ""

                if "overview" in heading_text or "background" in heading_text:
                    data["overview"] = content
                elif "uses" in heading_text or "effectiveness" in heading_text:
                    if content:
                        data["uses_effectiveness"].append(content)
                elif "safety" in heading_text and not data["safety_details"]:
                    data["safety_details"] = content
                elif "interact" in heading_text and not data["interactions_text"]:
                    data["interactions_text"] = content
                elif "dos" in heading_text and not data["dosage"]:
                    data["dosage"] = content
                elif "adverse" in heading_text or "side effect" in heading_text:
                    if content:
                        data["adverse_effects"].append(content)
                elif "pregnan" in heading_text and not data["pregnancy_safety"]:
                    data["pregnancy_safety"] = content
                elif "mechanism" in heading_text and not data["mechanism_of_action"]:
                    data["mechanism_of_action"] = content

        # ---- Brand names / Also known as ----
        for label in soup.find_all(["dt", "th", "strong", "b", "span"]):
            text = label.get_text(strip=True).lower()
            nxt = label.find_next_sibling()
            if not nxt:
                continue
            val = nxt.get_text(separator=", ", strip=True)
            if "brand" in text:
                data["brand_names"] = [s.strip() for s in val.split(",")]
            elif "also known" in text or "other name" in text:
                data["also_known_as"] = [s.strip() for s in val.split(",")]

        return data

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def main():
    # Load URL list
    with open("data/natmed_urls.json") as f:
        url_list = json.load(f)

    print(f"Total ingredients: {len(url_list)}")

    # Load existing progress
    results  = []
    done_urls = set()
    try:
        with open("data/natmed_supplements.json") as f:
            results = json.load(f)
        # Only count entries that actually have data
        results = [r for r in results if r.get("overview") or r.get("safety_details") or r.get("dosage")]
        done_urls = {r["url"] for r in results}
        url_list  = [u for u in url_list if u["url"] not in done_urls]
        print(f"▶️  Resuming — {len(results)} done with data, {len(url_list)} remaining\n")
    except FileNotFoundError:
        print("Starting fresh...\n")

    # Start browser and log in
    driver = create_driver()
    if not login(driver):
        return

    try:
        for i, item in enumerate(url_list):
            print(f"[{i+1}/{len(url_list)}] {item['name']}")

            result = scrape_page(driver, item["name"], item["url"])

            if result == "LOGIN_FAILED":
                print("Cannot recover login. Saving progress and stopping.")
                break

            if result:
                results.append(result)

            # Save every 20 items
            if (i + 1) % 20 == 0:
                with open("data/natmed_supplements.json", "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  💾 Progress saved ({len(results)} total)\n")

            time.sleep(1.5)

    finally:
        # Always save on exit even if crashed
        with open("data/natmed_supplements.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Done! {len(results)} ingredients scraped.")
        print("Output → data/natmed_supplements.json")
        driver.quit()

if __name__ == "__main__":
    main()