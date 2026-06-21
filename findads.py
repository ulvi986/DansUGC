import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SCRAPECREATORS_API_KEY")

def get_ads(page_id):

    if pd.isna(page_id):
        print("SKIP: page_id boşdur")
        return []

    page_id = str(page_id).strip()

    if page_id == "":
        print("SKIP: page_id empty string")
        return []

    url = "https://api.scrapecreators.com/v1/facebook/adLibrary/company/ads"

    headers = {
        "x-api-key": API_KEY
    }

    params = {
        "pageId": page_id
    }

    print(f"Searching ads for page_id={page_id}")

    try:
        r = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        print("STATUS:", r.status_code)

        if r.status_code != 200:
            print("ERROR RESPONSE:")
            print(r.text)
            return []

        data = r.json()

        print("SUCCESS")

        if "ads" in data:
            return data["ads"]

        if "results" in data:
            return data["results"]

        if "searchResults" in data:
            return data["searchResults"]

        print("Unknown response format")
        print(data)

        return []

    except Exception as e:
        print("EXCEPTION:", e)
        return []


companies = pd.read_csv("companies.csv")

all_ads = []

for _, company in companies.iterrows():

    page_id = company["page_id"]

    ads = get_ads(page_id)

    print("Found ads:", len(ads))

    for ad in ads:

        all_ads.append({
            "app_name": company["app_name"],
            "company_name": company["company_name"],
            "page_id": page_id,
            "ad_id": ad.get("id"),
            "start_date": ad.get("start_date")
        })

df = pd.DataFrame(all_ads)

df.to_csv("ads.csv", index=False)

print("TOTAL ADS:", len(df))
print("ads.csv yaradıldı")