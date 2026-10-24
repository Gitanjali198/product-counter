"""List the Gemini models your API key can use (so you can pick a valid GEMINI_MODEL).

  python list_models.py
"""
import os

from productcounter.gemini_counter import _load_dotenv


def main():
    _load_dotenv()
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY first (env var or .env file).")
    client = genai.Client(api_key=key)
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(m.name.replace("models/", ""))


if __name__ == "__main__":
    main()
