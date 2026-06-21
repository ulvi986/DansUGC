import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SCRAPECREATORS_API_KEY")


def search_appstore(keyword):
    url = "https://itunes.apple.com/search"
    params = {
        "term": keyword,
        "entity": "software",
        "country": "us",
        "limit": 20
    }

    res = requests.get(url, params=params)
    res.raise_for_status()

    apps = []

    for app in res.json()["results"]:
        apps.append({
            "app_name": app.get("trackName"),
            "developer": app.get("sellerName"),
            "app_url": app.get("trackViewUrl"),
            "rating": app.get("averageUserRating"),
            "category": app.get("primaryGenreName")
        })

    return apps


apps = search_appstore("AI journaling")
df = pd.DataFrame(apps)
df.to_csv("apps.csv", index=False)

print(df)