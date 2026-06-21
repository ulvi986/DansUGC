import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SCRAPECREATORS_API_KEY")

url = "https://api.scrapecreators.com/v1/facebook/adLibrary/company/ads"

headers = {
    "x-api-key": API_KEY
}

params = {
    "pageId": "1064882203382895"
}

r = requests.get(url, headers=headers, params=params, timeout=60)

print(r.status_code)
print(r.text)