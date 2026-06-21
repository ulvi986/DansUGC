"""
ads_winner_gemini.py
--------------------------------------------------------------------
Rəqib reklam datasını (ads.csv) analiz edir, "WINNER STRATEGY"
siqnallarını çıxarır və Gemini-yə YENİ, ORİJİNAL reklam strategiyası
yaratmaq üçün prompt qurub göndərir.

Qeyd: kopya yox -> Gemini-yə "yeni bir növ" winner strategy yaratmaq
tapşırığı verilir, rəqib reklamlar yalnız ilham/sübut kimi istifadə olunur.

Quraşdırma:
    pip install pandas google-genai
Açar:
    export GEMINI_API_KEY="sənin_açarın"
İşə salmaq:
    python ads_winner_gemini.py ads.csv
--------------------------------------------------------------------
"""

import os
import sys
import pandas as pd
from google import genai

import os
import sys
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()   # .env faylını oxuyur və GEMINI_API_KEY-i mühitə yükləyir
# ----------------------------------------------------------------------
# 1) DATANI OXU VƏ TƏMİZLƏ
# ----------------------------------------------------------------------
def load_ads(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # boş mətnləri at, tarixləri parse et
    df["ad_text"] = df["ad_text"].fillna("").astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    df["date_collected"] = pd.to_datetime(df["date_collected"], errors="coerce", utc=True)
    return df


# ----------------------------------------------------------------------
# 2) WINNER SİQNALLARINI ÇIXAR
#    Heç bir spend/impression metrikası olmadığı üçün ad-spy
#    evristikalarından istifadə edirik:
#      a) Eyni kreativ çox dəfə təkrarlanırsa -> advertiser onu
#         MİQYASLAYIR -> qazanan kreativ.
#      b) Köhnə start_date + hələ "active" -> uzun müddət dayanır ->
#         advertiser pul itirsəydi söndürərdi -> qazanan.
#      c) Hansı creative_type (video/şəkil) və platform üstünlük təşkil edir.
# ----------------------------------------------------------------------
def extract_winners(df: pd.DataFrame, top_n: int = 8) -> dict:
    today = df["date_collected"].max()

    # a) miqyaslanan kreativlər (eyni mətn neçə dəfə işlənir)
    scaled = (
        df[df["ad_text"].str.len() > 0]["ad_text"]
        .value_counts()
        .head(top_n)
    )

    # b) uzun müddət aktiv qalanlar (yaş = bu gün - start_date)
    active = df[df["status"].str.lower() == "active"].copy()
    active["days_running"] = (today - active["start_date"]).dt.days
    longest = (
        active.dropna(subset=["days_running"])
        .sort_values("days_running", ascending=False)
        [["advertiser_name", "ad_text", "creative_type", "days_running"]]
        .head(top_n)
    )

    return {
        "total_ads": len(df),
        "platform_mix": df["platform"].value_counts().to_dict(),
        "creative_mix": df["creative_type"].value_counts().to_dict(),
        "top_advertisers": df["advertiser_name"].fillna("(unknown)")
                              .value_counts().head(5).to_dict(),
        "scaled_creatives": scaled.to_dict(),          # winner siqnalı a
        "longest_running": longest.to_dict("records"), # winner siqnalı b
    }


# ----------------------------------------------------------------------
# 3) GEMINI ÜÇÜN PROMPT QUR
#    Promptu strukturlu (rol + kontekst + qaydalar + çıxış formatı) qururuq
#    ki, Gemini düzgün, yeni və orijinal strategiya versin.
# ----------------------------------------------------------------------
def build_prompt(app_name: str, winners: dict) -> str:
    scaled_block = "\n".join(
        f"  - {cnt}x repeated: {txt[:160]}"
        for txt, cnt in winners["scaled_creatives"].items()
    )
    longest_block = "\n".join(
        f"  - {r['days_running']} days active | {r['creative_type']} | "
        f"{r['advertiser_name']} | {str(r['ad_text'])[:120]}"
        for r in winners["longest_running"]
    )

    prompt = f"""You are an experienced performance-marketing strategist and creative director.

CONTEXT:
We analyzed competitors' live ads for the app "{app_name}".
The data below contains WINNER signals extracted from REAL ads.

OVERVIEW:
- Total ads: {winners['total_ads']}
- Platform breakdown: {winners['platform_mix']}
- Creative type breakdown: {winners['creative_mix']}
- Top advertisers: {winners['top_advertisers']}

SCALED CREATIVES (same copy used many times = winner = making money):
{scaled_block}

LONGEST-RUNNING ADS (not turned off = winner):
{longest_block}

TASK:
Use these winner signals as inspiration, but DO NOT COPY any of them.
Create a COMPLETELY NEW kind of winning ad strategy for "{app_name}".

RULES:
1. Do not reuse any existing copy — write original hooks and openers.
2. Account for which platform and creative type (video/image) dominates.
3. The strategy must be measurable and testable.

OUTPUT FORMAT (use exactly these headings):
1. STRATEGY NAME (short, memorable)
2. TARGET AUDIENCE & CORE PAIN POINT
3. 3 NEW HOOKS (for the first 3 seconds, original)
4. CREATIVE CONCEPT (video vs image, visual script)
5. PLATFORM PLAN (TikTok / Meta — why)
6. A/B TEST PLAN (what we will test)
"""
    return prompt

# ----------------------------------------------------------------------
# 4) GEMINI-YƏ GÖNDƏR
# ----------------------------------------------------------------------
def ask_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY mühit dəyişəni təyin edilməyib.")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "ads.csv"
    df = load_ads(csv_path)
    app_name = df["app_name"].mode().iloc[0] if not df["app_name"].isna().all() else "App"

    winners = extract_winners(df)
    prompt = build_prompt(app_name, winners)

    print("=" * 70)
    print("GEMINI-YƏ GÖNDƏRİLƏN PROMPT:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    # Açar yoxdursa, yalnız promptu göstərir (Gemini-yə göndərmir)
    if os.environ.get("GEMINI_API_KEY"):
        print("\nGEMINI-NİN CAVABI (yeni winner strategiya):\n")
        print(ask_gemini(prompt))
    else:
        print("\n[!] GEMINI_API_KEY yoxdur — yuxarıdakı promptu Gemini-yə əl ilə yapışdıra bilərsən.")


if __name__ == "__main__":
    main()