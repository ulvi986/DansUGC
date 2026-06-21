# Server Status

The Ad Intelligence Platform is now running at:
- **API**: http://localhost:8000
- **Frontend**: http://localhost:8000/app/

## How to Use
1. Open your browser and go to http://localhost:8000/app/
2. Type an app name in the input field (e.g., "Rosebud", "Solou", "Calm")
3. Click "Fetch ads & analyse" to get live ads and run the analysis.
4. The results will show winning patterns, strategy, scores, and agent outputs.

## Configuration
The backend is configured to use:
- **LLM**: OpenAI (if a valid `OPENAI_API_KEY` is provided in `.env`), otherwise falls back to heuristic mode.
- **Live Ad Fetching**: Enabled (using ScrapeCreators API key from `.env`).

## Notes
- The current `.env` contains a placeholder for the OpenAI key (`sk- your-openai-key-here`). 
  Please replace it with your actual OpenAI API key to enable LLM-powered analysis.
- The ScrapeCreators API key and Apify token are already present in `.env`.
- To restart the server after updating the `.env`, stop the current process and run:
  ```powershell
  cd E:\DansUGC\backend
  uvicorn main:app --reload --port 8000
  ```

## Features
- Live ad fetching for Meta and TikTok (via ScrapeCreators)
- Multi-agent analysis (text, video, image, features, patterns, scoring, voting, strategy)
- Evidence-based strategy generation
- History of past analyses
- Responsive dashboard

For any issues, check the server logs or contact support.