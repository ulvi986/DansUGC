import os
import time
import requests
import pandas as pd
from google import genai
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

df = pd.read_csv("ads.csv")
videos = df[df["creative_type"] == "video"].copy()

def download_video(url, path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

PROMPT = """
You are an ad intelligence analyst.

Analyze this video ad and extract winning creative features.

Use the ad text and video together.

Return only valid JSON:

{
  "hook": {
    "text": "",
    "type": "problem|curiosity|benefit|social_proof|fear|discount|story|other",
    "strength_score": 0
  },
  "creative_format": "ugc|demo|testimonial|founder_story|screen_recording|lifestyle|animation|other",
  "target_audience": "",
  "pain_point": "",
  "solution_angle": "",
  "emotional_trigger": "",
  "visual_features": {
    "human_present": true,
    "face_visible": true,
    "text_overlay": true,
    "app_screen_visible": true,
    "fast_cuts": true
  },
  "cta": "",
  "winner_strategy": "",
  "why_it_might_work": "",
  "winner_score": 0
}
"""

results = []

for i, row in videos.iterrows():
    video_path = f"video_{row['ad_id']}.mp4"

    try:
        download_video(row["image_or_video_url"], video_path)

        uploaded_file = client.files.upload(file=video_path)

        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                uploaded_file,
                f"""
App name: {row['app_name']}
Platform: {row['platform']}
Ad text: {row['ad_text']}
Country: {row['country']}

{PROMPT}
"""
            ]
        )

        results.append({
            "ad_id": row["ad_id"],
            "app_name": row["app_name"],
            "platform": row["platform"],
            "analysis": response.text
        })

    except Exception as e:
        results.append({
            "ad_id": row["ad_id"],
            "app_name": row["app_name"],
            "platform": row["platform"],
            "error": str(e)
        })

pd.DataFrame(results).to_csv("video_feature_analysis.csv", index=False)