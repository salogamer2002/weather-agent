"""
================================================================================
   Weather Agent - Flask Web Interface
   Original interactive experience on web browser
   Powered by Kimi K2 (Fireworks AI) + LangChain + OpenWeatherMap
================================================================================

DEPENDENCIES (install before running):
    pip install flask python-dotenv requests \
                langchain langchain-fireworks langchain-core \
                langgraph

REQUIRED .env keys:
    FIREWORKS_API_KEY=your_fireworks_key
    OPENWEATHER_API_KEY=your_openweathermap_key
    PORT=5000   (optional, defaults to 5000)
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK  (runs before anything else)
# ─────────────────────────────────────────────────────────────────────────────

import importlib
import importlib.util  # must be imported explicitly on Windows
import sys

REQUIRED = {
    "flask":               "flask",
    "dotenv":              "python-dotenv",
    "requests":            "requests",
    "langchain":           "langchain",
    "langchain_fireworks": "langchain-fireworks",
    "langchain_core":      "langchain-core",
    "langgraph":           "langgraph",
}

missing = []
for module, pkg in REQUIRED.items():
    if importlib.util.find_spec(module) is None:
        missing.append(pkg)

if missing:
    print("\n" + "="*60)
    print("  MISSING DEPENDENCIES — run the command below:")
    print("  pip install " + " ".join(missing))
    print("="*60 + "\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import logging
from collections import defaultdict
from typing import Tuple

from dotenv import load_dotenv
import requests
from flask import Flask, render_template_string, request, jsonify

# LangChain / LangGraph
from langchain_core.tools import tool
from langchain_fireworks import ChatFireworks
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

FIREWORKS_API_KEY   = os.getenv("FIREWORKS_API_KEY",   "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
PORT                = int(os.getenv("PORT", 5000))

if not FIREWORKS_API_KEY:
    print("ERROR: FIREWORKS_API_KEY not set in .env")
    sys.exit(1)

if not OPENWEATHER_API_KEY:
    print("ERROR: OPENWEATHER_API_KEY not set in .env")
    sys.exit(1)

OWM_BASE = "https://api.openweathermap.org/data/2.5"
OWM_GEO  = "http://api.openweathermap.org/geo/1.0"

# Kimi K2 on Fireworks — excellent function/tool calling support
KIMI_K2_MODEL = "accounts/fireworks/models/kimi-k2-instruct-0905"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("weather_agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("WeatherAgent")

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

class TopicValidator:
    """Validates weather-related questions."""

    WEATHER_KEYWORDS = {
        "weather", "temperature", "temp", "forecast", "rain", "snow", "cloud",
        "sunny", "wind", "humidity", "cold", "hot", "warm", "celsius", "fahrenheit",
        "climate", "storm", "coordinates", "location", "city", "compare",
    }
    NON_WEATHER_KEYWORDS = {
        "physics", "science", "math", "history", "programming", "code",
        "technology", "biology", "chemistry", "sports", "music", "recipe",
    }
    COMMON_CITIES = {
        "london", "paris", "tokyo", "new york", "dubai", "sydney", "delhi",
        "mumbai", "karachi", "shanghai", "moscow", "berlin", "toronto",
        "istanbul", "cairo", "bangkok", "singapore", "hong kong",
    }

    @staticmethod
    def is_weather_related(question: str) -> Tuple[bool, str]:
        q = question.lower()
        if any(kw in q for kw in TopicValidator.WEATHER_KEYWORDS):
            return True, ""
        if any(c in q for c in TopicValidator.COMMON_CITIES):
            return True, ""
        for kw in TopicValidator.NON_WEATHER_KEYWORDS:
            if kw in q:
                return False, f"ERROR: I only answer weather questions. '{kw}' is outside my scope."
        return False, "ERROR: Ask about weather, temperature, forecasts, or specific cities."


class InputValidator:
    """Validates / sanitises user input."""

    DANGEROUS_PATTERNS = [
        r"[;`|&$(){}<>\[\]]+",
        r"<script|javascript:",
        r"union\s+select|drop\s+table",
        r"\.\./|\.\.\\",
    ]

    @staticmethod
    def validate_city_name(city: str) -> Tuple[bool, str]:
        if not city or len(city) > 100:
            return False, ""
        city = city.strip()
        for p in InputValidator.DANGEROUS_PATTERNS:
            if re.search(p, city, re.IGNORECASE):
                return False, ""
        if not re.match(r"^[a-zA-Z\s\-']+$", city):
            return False, ""
        return True, city

    @staticmethod
    def validate_user_input(text: str) -> Tuple[bool, str]:
        if not text or len(text) > 1000:
            return False, "Input too long"
        for p in InputValidator.DANGEROUS_PATTERNS:
            if re.search(p, text, re.IGNORECASE):
                return False, "Suspicious pattern detected"
        return True, text


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER HELPERS  (pure functions — called by LangChain tools below)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_current_weather(city: str) -> str:
    ok, city = InputValidator.validate_city_name(city)
    if not ok:
        return "ERROR: Invalid city name"
    try:
        r = requests.get(
            f"{OWM_BASE}/weather",
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        return (
            f"\n{city.title()}, {d['sys']['country']}\n{'='*40}\n\n"
            f"Temperature:  {d['main']['temp']}C\n"
            f"Feels Like:   {d['main']['feels_like']}C\n"
            f"Condition:    {d['weather'][0]['description'].capitalize()}\n\n"
            f"Humidity:     {d['main']['humidity']}%\n"
            f"Wind Speed:   {d['wind']['speed']} m/s\n"
            f"{'='*40}\n"
        )
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return "ERROR: Could not fetch weather"


def _fetch_forecast(city: str, days: int = 3) -> str:
    ok, city = InputValidator.validate_city_name(city)
    if not ok:
        return "ERROR: Invalid city name"
    days = max(1, min(int(days), 5))
    try:
        r = requests.get(
            f"{OWM_BASE}/forecast",
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric", "cnt": days * 8},
            timeout=10,
        )
        r.raise_for_status()
        data  = r.json()
        daily = defaultdict(list)
        for item in data["list"]:
            daily[item["dt_txt"].split(" ")[0]].append(item)

        lines = [f"\n{city.title()}, {data['city']['country']} - {days} Day Forecast\n{'='*40}"]
        for date, entries in list(daily.items())[:days]:
            temps = [e["main"]["temp"] for e in entries]
            descs = [e["weather"][0]["description"] for e in entries]
            lines += [
                f"\n{date}",
                f"  Low:       {min(temps):.1f}C",
                f"  High:      {max(temps):.1f}C",
                f"  Condition: {max(set(descs), key=descs.count).capitalize()}",
            ]
        lines.append(f"\n{'='*40}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        return "ERROR: Could not fetch forecast"


def _fetch_coordinates(city: str) -> str:
    ok, city = InputValidator.validate_city_name(city)
    if not ok:
        return "ERROR: Invalid city name"
    try:
        r = requests.get(
            f"{OWM_GEO}/direct",
            params={"q": city, "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return "ERROR: City not found"
        p = data[0]
        return (
            f"\n{p.get('name')}, {p.get('country')}\n{'='*40}\n\n"
            f"Latitude:     {p['lat']}\n"
            f"Longitude:    {p['lon']}\n\n{'='*40}\n"
        )
    except Exception as e:
        logger.error(f"Coordinates failed: {e}")
        return "ERROR: Could not fetch coordinates"


def _fetch_compare(city1: str, city2: str) -> str:
    ok1, c1 = InputValidator.validate_city_name(city1)
    ok2, c2 = InputValidator.validate_city_name(city2)
    if not (ok1 and ok2):
        return "ERROR: Invalid city names"
    try:
        def fetch(c):
            resp = requests.get(
                f"{OWM_BASE}/weather",
                params={"q": c, "appid": OPENWEATHER_API_KEY, "units": "metric"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

        d1, d2 = fetch(c1), fetch(c2)

        def row(d):
            return {
                "name":     f"{d['name']}, {d['sys']['country']}",
                "temp":     d["main"]["temp"],
                "humidity": d["main"]["humidity"],
                "desc":     d["weather"][0]["description"].capitalize(),
                "wind":     d["wind"]["speed"],
            }

        r1, r2 = row(d1), row(d2)
        return (
            f"\nWEATHER COMPARISON\n{'='*50}\n\n"
            f"{r1['name']:<25} vs  {r2['name']}\n{'-'*50}\n"
            f"Temperature:  {r1['temp']}C{'':<15}{r2['temp']}C\n"
            f"Humidity:     {r1['humidity']}%{'':<16}{r2['humidity']}%\n"
            f"Condition:    {r1['desc']:<21}{r2['desc']}\n"
            f"Wind Speed:   {r1['wind']} m/s{'':<13}{r2['wind']} m/s\n"
            f"\n{'='*50}\n"
        )
    except Exception as e:
        logger.error(f"Compare failed: {e}")
        return "ERROR: Could not compare"


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN TOOLS  (Kimi K2 reads docstrings to decide which to call)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_current_weather(city: str) -> str:
    """
    Get the current weather conditions for a city.

    Args:
        city: The name of the city (e.g. 'London', 'Karachi', 'Tokyo').

    Returns:
        Temperature, feels-like, condition, humidity, and wind speed.
    """
    logger.info(f"[Tool] get_current_weather -> {city}")
    return _fetch_current_weather(city)


@tool
def get_weather_forecast(city: str, days: int = 3) -> str:
    """
    Get the weather forecast for a city over the next 1-5 days.

    Args:
        city: The name of the city.
        days: Number of forecast days (1-5). Defaults to 3.

    Returns:
        Daily high/low temperatures and conditions for each forecast day.
    """
    logger.info(f"[Tool] get_weather_forecast -> {city}, {days} days")
    return _fetch_forecast(city, days)


@tool
def get_city_coordinates(city: str) -> str:
    """
    Look up the geographic coordinates (latitude and longitude) of a city.

    Args:
        city: The name of the city.

    Returns:
        Latitude and longitude of the city.
    """
    logger.info(f"[Tool] get_city_coordinates -> {city}")
    return _fetch_coordinates(city)


@tool
def compare_weather(city1: str, city2: str) -> str:
    """
    Compare the current weather between two cities side-by-side.

    Args:
        city1: First city name.
        city2: Second city name.

    Returns:
        Side-by-side comparison of temperature, humidity, condition, and wind.
    """
    logger.info(f"[Tool] compare_weather -> {city1} vs {city2}")
    return _fetch_compare(city1, city2)


TOOLS = [get_current_weather, get_weather_forecast, get_city_coordinates, compare_weather]

# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN REACT AGENT  (Kimi K2 via Fireworks)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a friendly and precise weather assistant powered by Kimi K2.
You ONLY answer questions related to weather, forecasts, city coordinates,
and weather comparisons. For anything unrelated, politely decline.

Rules:
- Always call the appropriate tool to get real-time data. Never guess or invent numbers.
- Respond concisely and format the data clearly for the user.
- If a city is not found, say so and ask the user to check the spelling.
"""

_llm = ChatFireworks(
    model=KIMI_K2_MODEL,
    temperature=0,
    fireworks_api_key=FIREWORKS_API_KEY,
    max_tokens=1024,
)

_agent = create_react_agent(_llm, TOOLS, state_modifier=SYSTEM_PROMPT)
logger.info(f"[LangChain] ReAct agent ready — {KIMI_K2_MODEL} + {[t.name for t in TOOLS]}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class WeatherAgent:
    """Thin wrapper around the LangChain ReAct agent."""

    def run(self, question: str) -> str:
        try:
            result   = _agent.invoke({"messages": [HumanMessage(content=question)]})
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    return msg.content
            return "ERROR: No response from agent"
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"ERROR: {e}"


agent = WeatherAgent()

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Weather Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container { max-width: 800px; margin: 0 auto; }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            padding: 20px;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p  { font-size: 1.1em; opacity: 0.9; }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.82em;
            margin-top: 8px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }

        .input-section { display: flex; gap: 10px; margin-bottom: 20px; }

        #question {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        #question:focus { outline: none; border-color: #667eea; }

        button {
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover  { background: #764ba2; }
        button:active { transform: scale(0.98); }

        .output {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.6;
            color: #333;
            max-height: 500px;
            overflow-y: auto;
        }

        .error {
            color: #d32f2f;
            background: #ffebee;
            border-left: 4px solid #d32f2f;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }

        .thinking {
            color: #667eea;
            background: #f0f0ff;
            border-left: 4px solid #667eea;
            padding: 12px 15px;
            border-radius: 4px;
            margin: 15px 0;
            font-style: italic;
        }

        .examples { margin-top: 20px; }
        .examples h3 { color: #667eea; margin-bottom: 10px; }
        .examples button {
            display: block; width: 100%; text-align: left;
            margin: 8px 0; padding: 10px 15px;
            background: #f0f0f0; color: #333; border: 1px solid #ddd;
        }
        .examples button:hover { background: #e0e0e0; }
        .input-hint { color: #999; font-size: 0.9em; margin-top: 5px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🌤️ Weather Agent</h1>
        <p>Ask about weather, forecasts, coordinates, and more!</p>
        <div class="badge">⚡ Kimi K2 · LangChain · OpenWeatherMap</div>
    </div>

    <div class="card">
        <h2 style="margin-bottom: 20px;">Ask a Weather Question</h2>

        <div class="input-section">
            <input type="text" id="question"
                   placeholder="E.g., What is the weather in Karachi?"
                   autocomplete="off">
            <button onclick="askQuestion()">Ask</button>
        </div>
        <div class="input-hint">💡 Ask about weather, forecasts, coordinates, or compare cities</div>

        <div id="output"></div>

        <div class="examples">
            <h3>Example Questions:</h3>
            <button onclick="setQ('What is the weather in Karachi?')">Current weather in Karachi</button>
            <button onclick="setQ('Give me a 5-day forecast for London')">5-day forecast for London</button>
            <button onclick="setQ('What are the coordinates of Tokyo?')">Coordinates of Tokyo</button>
            <button onclick="setQ('Compare weather between Dubai and Istanbul')">Compare Dubai vs Istanbul</button>
        </div>
    </div>
</div>

<script>
    function setQ(text) {
        document.getElementById('question').value = text;
        askQuestion();
    }

    function askQuestion() {
        const question = document.getElementById('question').value.trim();
        if (!question) { showOutput('ERROR: Please enter a question', true); return; }

        document.getElementById('output').innerHTML =
            '<div class="thinking">🤔 Kimi K2 is thinking...</div>';

        fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) showOutput(d.error, true);
            else         showOutput(d.response, false);
        })
        .catch(e => showOutput('ERROR: ' + e.message, true));
    }

    function showOutput(text, isError) {
        const out = document.getElementById('output');
        out.innerHTML = isError
            ? `<div class="error">${text}</div>`
            : `<div class="output">${text}</div>`;
    }

    document.getElementById('question').addEventListener('keypress', e => {
        if (e.key === 'Enter') askQuestion();
    });
    document.getElementById('question').focus();
</script>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data     = request.json
    question = (data or {}).get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    is_weather, msg = TopicValidator.is_weather_related(question)
    if not is_weather:
        return jsonify({"error": msg}), 400

    is_valid, msg = InputValidator.validate_user_input(question)
    if not is_valid:
        return jsonify({"error": f"ERROR: {msg}"}), 400

    response = agent.run(question)
    logger.info(f"Q: {question[:60]}...")
    return jsonify({"question": question, "response": response})


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model":  KIMI_K2_MODEL,
        "tools":  [t.name for t in TOOLS],
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  WEATHER AGENT — Flask + LangChain + Kimi K2 (Fireworks)")
    print("="*70)
    print(f"[OK] FIREWORKS_API_KEY configured")
    print(f"[OK] OPENWEATHER_API_KEY configured")
    print(f"[OK] Model: {KIMI_K2_MODEL}")
    print(f"[OK] LangChain tools: {[t.name for t in TOOLS]}")
    print(f"[OK] Guard rails active")
    print(f"[OK] Starting on http://0.0.0.0:{PORT}")
    print("="*70 + "\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)