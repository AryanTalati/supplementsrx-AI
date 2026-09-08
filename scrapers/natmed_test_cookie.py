# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:41:18 2026

@author: priya
"""

import requests
from bs4 import BeautifulSoup

# Paste your FRESH cookie here
COOKIE_STRING = "AUTH_SESSION_ID=YTA5ZjU3MjYtN2I1OC00MDI2LWI1YjktNjdkNGFmZTFmNzdlLkhzS0ZXZVpTamUySUR2b2VtdlZtRzU5UlpVN181X1lnd1ZCWkljX3F1LXhxRk9XUVlYVFBuNF83dFk5UmxpR1lXR3JRRjRfZlptX3N2OTY4SkpvLVFn.localhost-19201; KC_AUTH_SESSION_HASH=eOtKmxPgZDAJKNj7mLLM1yXVTY1NF+cLtG3ME40KNvs; KC_RESTART=eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..OvquIZkDZIaWuTJKUMS1Zw.mq3NdM2GFxZJhWdQbAF8pI08QW1IJyztjG6jnR30DsN99eIVAVemEJCaZPFzm4bbqSwjmWehK68nRIb4Ap7Ws62qEYU9RqtKu9hzkUKr0WSW1lPnQ66K8TY3FMrVcuHQ87BXJiEKEzk62Uu75c3oOK-Iqc07ae4i1TlDf-XC0INSXKaHn91fFZLEKX8ScZTSQVG12XNhD5zOSWsDiTJyThenvugA1pqBsrv1E1gGk5UjjADxbvxeyxFzWk7089TlnGj3an587KLt3qHwg0lTloKSMjEQ59yHvpeJvMtokIpUoUeyr-Kh-xrQBXvoXIBRasoZ_cf37FsGEo7oB0bmErMN_aWiQ0wl1k97BNSBneA1MdYdi6Sxzzn8bx4FbOUNj9ZrRprt2OHi6FuQeMKr_NzEAlYNC9hTQlXqFuOwvkhck3TEyv21cBFABg-MDDLpXldEAYePOE3FtSG1yQllO1W2ziPUQaRyNgRAOsuyvV-YkOnLDdoTREmVdHSCI_m9fXXmjnXNi_CUmKbovenNoCeLyRu5UWlNufTVI2sIOpVQ7Pt7t1TEOrLLmPs7v5BQxppxyuiSwtCtSPmDSpiGtKNDu_9EU2OfF6My9Zon2MPCBMRO_Iax7QRpRzeDUVbuFa0Y6e-3je9bf7k4GwZj3Wem56JzGn58T8CCJV5fgsntoeDZvcI2kbfu3zNXQ_gSWizwJfO9whBAPUsh-qpp1Qund0ou7Zz2MOxID0sHO_VZZQSyoMxvjsllVyX_almARdew8Z2DaX2A30z6dRLcvawXyfFJI0_yIH-Hd7ro0bewk6XKblvY2_8EHIpXCYO3kNg9Hk9oZfhAwQCe791xLzcbihcqrClLbYosAPmvEtrk6xZEv-oWEgdUgxbrljx5gzi7CK6U0f_xfClJLzKrQ1YdbD8UUwF0Iw8F8R2R5Vo75PmzXQx5jnwgx73rAAqgkfDf4TLW0H0HN8JfHQS1NbECUftAFwiuaoP6prhhlOvT13SBHF-bsJht8SM-GeDeyO_8t1VKbAzwvUhdagfIjWqbgqceG3_X0upT9oRAZt80XF8uO3QdA-U_0hXRNGbbOx0L9Mw887MyTqq_bxGAxoEgDbSq7UK_LHORiSK1UwloxnALZVPBokdX5ZEWKyo_N7O__2-ydtnftAo_3yYX-L6BeKBudAfE18VX4WiBAmunSZPJSkh80gNXIP4p28IR.d6-uRUqOuiq0JqC8jyf77g; optimizelyEndUserId=oeu1770952830695r0.5435717326389227; LetterSite=naturalmedicines.therapeuticresearch.com; LastProductAccessed=naturalmedicines.therapeuticresearch.com; product=naturalmedicines.therapeuticresearch.com; _gid=GA1.2.53251108.1773789325; trc_sc_ceid=CE50710117; trcauth=True; _dc_gtm_UA-1428672-3=1; _ga=GA1.1.1806443433.1770952831; _ga_89300NMX1V=GS2.1.s1773801371$o7$g1$t1773801387$j44$l0$h0; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Mar+17+2026+19%3A36%3A27+GMT-0700+(Mountain+Standard+Time)&version=202506.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=86a940ef-d476-4dbf-89b2-073c28e20b53&interactionCount=0&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0004%3A0%2CC0003%3A1&AwaitingReconsent=false"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Cookie": COOKIE_STRING
}

url = "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Ashwagandha"
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

title = soup.title.text if soup.title else "No title"
print(f"Page title: {title}")

if "login" in title.lower():
    print()
    print("❌ COOKIE FAILED — still redirecting to login page")
    print("   Try copying the cookie again from browser")
else:
    print()
    print("✅ COOKIE WORKS — successfully logged in!")
    print("   You can now run natmed_scraper.py")
    print()
    # Show all IDs so we can see real page structure
    print("=== IDs on real page ===")
    for tag in soup.find_all(id=True):
        print(f"  id={tag['id']}  tag={tag.name}")
    print()
    print("=== Headings ===")
    for tag in soup.find_all(["h1","h2","h3"]):
        print(f"  {tag.name}: {tag.get_text(strip=True)[:80]}")