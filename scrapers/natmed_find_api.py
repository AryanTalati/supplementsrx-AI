# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:56:02 2026

@author: priya
"""
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

USERNAME = "hdavulcu@asu.edu"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

LOGIN_URL = "https://naturalmedicines.therapeuticresearch.com/api/sitecore/account/SingleSignOn/?url=%2fHome%2fND"
TEST_URL  = "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Ashwagandha"

# Enable network logging to capture API calls
options = webdriver.ChromeOptions()
options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
options.add_argument("--window-size=1920,1080")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# Login
print("Logging in...")
driver.get(LOGIN_URL)
wait = WebDriverWait(driver, 20)
wait.until(EC.element_to_be_clickable((By.ID, "username"))).send_keys(USERNAME)
wait.until(EC.element_to_be_clickable((By.ID, "password"))).send_keys(PASSWORD)
wait.until(EC.element_to_be_clickable((By.ID, "kc-login"))).click()
time.sleep(5)
print(f"Login status: {driver.current_url}")

# Clear logs before visiting the page
driver.get_log("performance")

# Visit Ashwagandha page
print("\nOpening Ashwagandha page...")
driver.get(TEST_URL)
time.sleep(6)  # Wait for all API calls to complete

# Capture all network requests
print("\n=== ALL API/NETWORK CALLS MADE BY THIS PAGE ===")
logs = driver.get_log("performance")
api_calls = []

for log in logs:
    try:
        msg = json.loads(log["message"])["message"]
        if msg["method"] == "Network.responseReceived":
            url = msg["params"]["response"]["url"]
            mime = msg["params"]["response"]["mimeType"]
            status = msg["params"]["response"]["status"]
            # Only show API/data calls (not images, css, fonts)
            if any(x in url for x in ["api", "data", "json", "monograph", "supplement"]):
                print(f"\n  URL: {url}")
                print(f"  Type: {mime} | Status: {status}")
                api_calls.append(url)
            elif "json" in mime.lower():
                print(f"\n  JSON URL: {url}")
                print(f"  Status: {status}")
                api_calls.append(url)
    except:
        pass

print(f"\n\nTotal API calls found: {len(api_calls)}")

# Save findings
with open("scrapers/data/api_calls.txt", "w") as f:
    f.write("\n".join(api_calls))
print("Saved to scrapers/data/api_calls.txt")

input("\nPress Enter to close...")
driver.quit()