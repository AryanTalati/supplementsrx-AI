# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:47:19 2026

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
import time

USERNAME = "hdavulcu@asu.edu"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

LOGIN_URL = "https://naturalmedicines.therapeuticresearch.com/api/sitecore/account/SingleSignOn/?url=%2fHome%2fND"
TEST_URL  = "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Ashwagandha"

def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver

driver = create_driver()

# Login
print("Logging in...")
driver.get(LOGIN_URL)
wait = WebDriverWait(driver, 20)
wait.until(EC.element_to_be_clickable((By.ID, "username"))).send_keys(USERNAME)
wait.until(EC.element_to_be_clickable((By.ID, "password"))).send_keys(PASSWORD)
wait.until(EC.element_to_be_clickable((By.ID, "kc-login"))).click()
time.sleep(5)
print(f"After login URL: {driver.current_url}")

# Go to Ashwagandha page
print("\nOpening Ashwagandha page...")
driver.get(TEST_URL)

# Wait different amounts and check content each time
for wait_time in [2, 4, 6, 10]:
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    body_text = soup.get_text(strip=True)
    print(f"\n--- After {wait_time}s total wait ---")
    print(f"Page title: {soup.title.text if soup.title else 'None'}")
    print(f"Body text length: {len(body_text)} chars")
    print(f"First 300 chars of body: {body_text[:300]}")
    print(f"All IDs: {[t['id'] for t in soup.find_all(id=True)]}")

# Save full HTML to inspect
with open("scrapers/data/ashwagandha_raw.html", "w", encoding="utf-8") as f:
    f.write(driver.page_source)
print("\n💾 Full HTML saved to scrapers/data/ashwagandha_raw.html")
print("Open this file in Chrome to see what was actually captured")

# Also print ALL text content
print("\n=== ALL TEXT ON PAGE ===")
soup = BeautifulSoup(driver.page_source, "html.parser")
for tag in soup.find_all(["nav","header","footer","script","style"]):
    tag.decompose()
print(soup.get_text(separator="\n", strip=True)[:3000])

input("\nPress Enter to close browser...")
driver.quit()