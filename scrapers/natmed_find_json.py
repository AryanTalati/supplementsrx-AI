# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:05:01 2026

@author: priya
"""
import os

# scrapers/natmed_find_json.py
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

options = webdriver.ChromeOptions()
options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
options.add_argument("--window-size=1920,1080")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Login
print("Logging in...")
driver.get(LOGIN_URL)
wait = WebDriverWait(driver, 20)
wait.until(EC.element_to_be_clickable((By.ID, "username"))).send_keys(USERNAME)
wait.until(EC.element_to_be_clickable((By.ID, "password"))).send_keys(PASSWORD)
wait.until(EC.element_to_be_clickable((By.ID, "kc-login"))).click()
time.sleep(5)

# Clear old logs
driver.get_log("performance")

# Visit page
print("Opening Ashwagandha...")
driver.get(TEST_URL)
time.sleep(8)

# Capture ALL network requests this time — not just API ones
print("\n=== ALL NETWORK REQUESTS ===")
logs = driver.get_log("performance")

all_urls = []
for log in logs:
    try:
        msg = json.loads(log["message"])["message"]
        if msg["method"] == "Network.responseReceived":
            url   = msg["params"]["response"]["url"]
            mime  = msg["params"]["response"].get("mimeType", "")
            status = msg["params"]["response"]["status"]
            all_urls.append(f"{status} | {mime[:30]} | {url}")
    except:
        pass

# Print everything — we want to see ALL calls
for u in all_urls:
    print(u)

# Save to file
with open("scrapers/data/all_network_calls.txt", "w") as f:
    f.write("\n".join(all_urls))

print(f"\nTotal: {len(all_urls)} requests")
print("Saved to scrapers/data/all_network_calls.txt")

input("Press Enter to close...")
driver.quit()