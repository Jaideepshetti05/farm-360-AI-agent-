import os
import requests

class WeatherClient:
    """OpenWeather API integration."""
    def __init__(self):
        self.api_key = os.environ.get("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_forecast(self, location, days=3):
        if not self.api_key:
            # Fallback to logic if no API key is set
            return {
                "location": location,
                "current_temp_c": 28,
                "rain_chance": 85,
                "humidity": 60,
                "note": "Mocked (No API key)"
            }

        try:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "location": data.get("name", location),
                "current_temp_c": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rain_chance": 100 if "rain" in data else 0, # Simple heuristic for current weather
                "description": data["weather"][0]["description"] if data.get("weather") else ""
            }
        except Exception as e:
            return {"error": str(e), "note": "Failed to fetch real data"}
