import os
import pandas as pd
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
client = ApifyClient(APIFY_TOKEN)

ACTOR_ID = "apify/facebook-ads-scraper"

companies = pd.read_csv("companies.csv")

rows = []

for _, company in companies.iterrows():
    page_id = str(company["page_id"]).strip()

    if page_id == "" or page_id.lower() == "nan":
        print("SKIP empty page_id")
        continue

    meta_url = (
        "https://www.facebook.com/ads/library/"
        f"?active_status=active&ad_type=all&country=US"
        f"&search_type=page&view_all_page_id={page_id}"
    )

    print("Running Apify for:", company["company_name"])
    print(meta_url)

    run_input = {
        "startUrls": [
            {
                "url": meta_url
            }
        ],
        "maxItems": 20
    }

    try:
        run = client.actor(ACTOR_ID).call(run_input=run_input)

        dataset_id = run["defaultDatasetId"]
        items = client.dataset(dataset_id).list_items().items

        print("Found:", len(items))

        for ad in items:
            rows.append({
                "app_name": company["app_name"],
                "company_name": company["company_name"],
                "page_id": page_id,
                "ad_id": ad.get("adArchiveId") or ad.get("id"),
                "ad_text": ad.get("adText") or ad.get("text"),
                "start_date": ad.get("startDate"),
                "end_date": ad.get("endDate"),
                "page_name": ad.get("pageName"),
                "url": ad.get("url")
            })

    except Exception as e:
        print("ERROR:", e)

df = pd.DataFrame(rows)
df.to_csv("ads.csv", index=False)

print("TOTAL ADS:", len(df))
print("ads.csv yaradıldı")