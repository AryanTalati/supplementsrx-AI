# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:44:01 2026

@author: priya
"""

import requests
from bs4 import BeautifulSoup
import json
import time

# ============================================================
#  PASTE YOUR FULL COOKIE STRING BETWEEN THE QUOTES BELOW
#  Example: ".ASPXAUTH=AbCdEf123...; ASP.NET_SessionId=xyz..."
# ============================================================
COOKIE_STRING = "shell#lang=en; LetterSite=naturalmedicines.therapeuticresearch.com; LastProductAccessed=naturalmedicines.therapeuticresearch.com; product=naturalmedicines.therapeuticresearch.com; _pk_id.47.397e=2195f098209f13d5.1773688608.; optimizelyEndUserId=oeu1773688608061r0.8995914686351482; cebs=1; hubspotutk=b8005101df42232b3bff956425758d14; __hssrc=1; _hjSessionUser_682196=eyJpZCI6Ijc3NmY4MmYwLTk4YTQtNWIyZS04ODc4LTk5NzBmYjI3MmQxYSIsImNyZWF0ZWQiOjE3NzM2ODg2MDg0MjYsImV4aXN0aW5nIjp0cnVlfQ==; IPReferer=IpAddress=64.234.124.2&IpAddressKnown=false&QsReferrer=&QsReferrerKnown=false; _gid=GA1.2.770562506.1773790142; _hjSession_682196=eyJpZCI6IjI4ZTM3MjY3LWZiMzAtNDhkMC1iOTVjLTQ4NGQ2NjBjNGViZCIsImMiOjE3NzM3OTAxNDI1ODgsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; _pk_ref.47.397e=%5B%22%22%2C%22%22%2C1773790143%2C%22https%3A%2F%2Fwww.google.com%2F%22%5D; _pk_ses.47.397e=1; _ce.clock_data=-319%2C64.234.124.2%2C1%2C7c73ef5b8d3235ae0606f2e84e457ff5%2CChrome%2CUS; _ce.s=v~1b707ce88e4fc246365e03233771ca9ec809b629~lcw~1773790143075~vir~returning~lva~1773790142836~vpv~0~v11ls~155790d0-2259-11f1-a19a-f5faefd70c84~v11.cs~469824~v11.s~155790d0-2259-11f1-a19a-f5faefd70c84~v11.vs~1b707ce88e4fc246365e03233771ca9ec809b629~v11.fsvd~eyJ1cmwiOiJuYXR1cmFsbWVkaWNpbmVzLnRoZXJhcGV1dGljcmVzZWFyY2guY29tL2hvbWUvbmQiLCJyZWYiOiIiLCJ1dG0iOltdfQ%3D%3D~v11.sla~1773790143074~v11.wss~1773790143075~lcw~1773790143077; __hstc=254013463.b8005101df42232b3bff956425758d14.1773688608931.1773688608932.1773790143879.2; ASP.NET_SessionId=muvmq4ol0qb13w5viw1u0fg5; trc_sc_ceid=CE50710117; .ASPXAUTH=409689C90724932065FF121A9796FC3D31175BA461BD7D885A2BFE2508DB7F6510CBB605D27E444B33D05B58CE5E31E9EF244D22CE8B8C21AF1DCC97B6598EC1B5C00D051DE4F95D176BBB824473A7AC; trcauth=True; OptanonAlertBoxClosed=2026-03-17T23:32:31.512Z; _gcl_au=1.1.717293931.1773790352; _fbp=fb.1.1773790351912.11534775750051400; _clck=19rocy%5E2%5Eg4f%5E0%5E2267; _ga_89300NMX1V=GS2.1.s1773790142$o2$g1$t1773790387$j59$l0$h0; _ga=GA1.2.1024237785.1773688608; _dc_gtm_UA-1428672-3=1; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Mar+17+2026+16%3A33%3A07+GMT-0700+(Mountain+Standard+Time)&version=202506.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=d36a18d6-a70e-4c78-a4d2-dc937f5d5d22&interactionCount=2&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0004%3A1%2CC0003%3A1&AwaitingReconsent=false&intType=1&geolocation=US%3BAZ; __hssc=254013463.5.1773790143879; _rdt_uuid=1773688608009.ef727816-5b70-42fc-9be5-cb3cf2b60132; cebsp_=9; _uetsid=91b60820225911f18c16ff645d3a0611; _uetvid=91b638f0225911f19234076af423f5f9; _clsk=1ishyd6%5E1773790388182%5E2%5E1%5Ei.clarity.ms%2Fcollect"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Cookie": COOKIE_STRING,
    "Referer": "https://naturalmedicines.therapeuticresearch.com/"
}

def scrape_monograph(name, url):
    """Scrape every data field from one ingredient page"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        # Check if session has expired (redirected to login)
        if "login" in response.url.lower() or response.status_code == 401:
            print("⚠️  Cookie expired! Re-copy from browser and update COOKIE_STRING.")
            return "SESSION_EXPIRED"
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # ---- Helper: safely get text from a section ----
        def get_section_text(section_id):
            el = soup.find(id=section_id) or \
                 soup.find("div", {"data-section": section_id}) or \
                 soup.find("section", {"id": section_id})
            return el.get_text(separator=" ", strip=True) if el else ""

        def get_section_list(section_id):
            el = soup.find(id=section_id)
            if not el:
                return []
            return [li.get_text(strip=True) for li in el.find_all("li")]

        # ---- Build the data dictionary ----
        data = {
            "name":               name,
            "url":                url,
            "source":             "NaturalMedicines",

            # All the fields we want
            "overview":           get_section_text("overview"),
            "uses_effectiveness": get_section_list("uses"),
            "safety_rating":      "",   # filled below
            "safety_details":     get_section_text("safety"),
            "interactions":       get_section_list("interactions"),
            "interactions_text":  get_section_text("interactions"),
            "dosage":             get_section_text("dosing"),
            "adverse_effects":    get_section_list("adverse-effects"),
            "pregnancy_safety":   get_section_text("pregnancy"),
            "mechanism_of_action": get_section_text("mechanism-of-action"),
            "brand_names":        [],
            "also_known_as":      []
        }

        # ---- Safety Rating (e.g. "Likely Safe") ----
        # Usually displayed as a badge/label near the top
        safety_badge = soup.find("span", class_=lambda c: c and "safety" in c.lower()) or \
                       soup.find("div",  class_=lambda c: c and "safety-rating" in c.lower())
        if safety_badge:
            data["safety_rating"] = safety_badge.get_text(strip=True)

        # ---- Brand Names / Also Known As ----
        for label in soup.find_all(["dt", "th", "strong", "b"]):
            text = label.get_text(strip=True).lower()
            sibling = label.find_next_sibling()
            if not sibling:
                continue
            sibling_text = sibling.get_text(separator=", ", strip=True)

            if "brand" in text:
                data["brand_names"] = [s.strip() for s in sibling_text.split(",")]
            elif "also known" in text or "other name" in text or "common name" in text:
                data["also_known_as"] = [s.strip() for s in sibling_text.split(",")]

        # ---- Fallback: if sections not found by ID, try heading text ----
        if not data["overview"]:
            for tag in soup.find_all(["h2", "h3"]):
                heading = tag.get_text(strip=True).lower()
                content_div = tag.find_next_sibling("div") or tag.find_parent("div")
                if not content_div:
                    continue
                content = content_div.get_text(separator=" ", strip=True)

                if "overview" in heading or "background" in heading:
                    data["overview"] = content
                elif "uses" in heading or "effectiveness" in heading:
                    data["uses_effectiveness"].append(content)
                elif "safety" in heading:
                    data["safety_details"] = content
                elif "interact" in heading:
                    data["interactions_text"] = content
                elif "dos" in heading:
                    data["dosage"] = content
                elif "adverse" in heading or "side effect" in heading:
                    data["adverse_effects"].append(content)
                elif "pregnan" in heading:
                    data["pregnancy_safety"] = content
                elif "mechanism" in heading:
                    data["mechanism_of_action"] = content

        return data

    except Exception as e:
        print(f"  ❌ Error scraping {name}: {e}")
        return None


def main():
    # Load URL list from Step 3
    with open("data/natmed_urls.json") as f:
        url_list = json.load(f)

    print(f"Total ingredients to scrape: {len(url_list)}")

    results = []
    failed  = []

    # ---- Auto-resume if script was interrupted ----
    try:
        with open("data/natmed_supplements.json") as f:
            results = json.load(f)
        done_urls = {r["url"] for r in results}
        url_list  = [u for u in url_list if u["url"] not in done_urls]
        print(f"▶️  Resuming — {len(results)} done, {len(url_list)} remaining\n")
    except FileNotFoundError:
        print("Starting fresh...\n")

    for i, item in enumerate(url_list):
        print(f"[{i+1}/{len(url_list)}] Scraping: {item['name']}")
        result = scrape_monograph(item["name"], item["url"])

        if result == "SESSION_EXPIRED":
            # Save progress and stop cleanly
            with open("data/natmed_supplements.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Progress saved — {len(results)} scraped so far.")
            print("Fix the cookie in COOKIE_STRING then re-run to continue.")
            return

        if result is None:
            failed.append(item)
        else:
            results.append(result)

        # Save progress every 20 items
        if len(results) % 20 == 0:
            with open("data/natmed_supplements.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  💾 Progress saved ({len(results)} total)\n")

        time.sleep(1.5)  # Wait between requests — important!

    # Final saves
    with open("data/natmed_supplements.json", "w") as f:
        json.dump(results, f, indent=2)

    if failed:
        with open("data/natmed_failed.json", "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\n⚠️  {len(failed)} failed — see data/natmed_failed.json")

    print(f"\n✅ Done! {len(results)} ingredients scraped.")
    print("Output → data/natmed_supplements.json")

if __name__ == "__main__":
    main()