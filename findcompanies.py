import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SCRAPECREATORS_API_KEY")

def search_company(query):
    url = "https://api.scrapecreators.com/v1/facebook/adLibrary/search/companies"

    headers = {
        "x-api-key": API_KEY
    }

    params = {
        "query": query
    }

    r = requests.get(url, headers=headers, params=params)
    print("Searching:", query, r.status_code)

    if r.status_code != 200:
        print("ERROR:", r.text)
        return []

    data = r.json()
    return data.get("searchResults", [])

apps = pd.read_csv("apps.csv")

rows = []

for _, app in apps.iterrows():
    app_name = app["app_name"]

    companies = search_company(app_name)

    for c in companies:
        rows.append({
            "app_name": app_name,
            "company_name": c.get("name"),
            "page_id": c.get("page_id"),
            "category": c.get("category"),
            "likes": c.get("likes"),
            "verification": c.get("verification"),
            "entity_type": c.get("entity_type")
        })

df = pd.DataFrame(rows)
df.to_csv("companies.csv", index=False)

print(df)
print("companies.csv yaradıldı")