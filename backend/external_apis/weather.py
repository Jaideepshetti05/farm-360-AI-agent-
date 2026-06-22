import os
import requests
from loguru import logger


class WeatherClient:
    """OpenWeather API integration with connection pooling via requests.Session."""
    def __init__(self):
        self.api_key = os.environ.get("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self._session = None

    def _get_session(self) -> requests.Session:
        """Get or create a shared HTTP session (connection pooling)."""
        if self._session is None:
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=5,
                pool_maxsize=10,
                max_retries=2,
            )
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        return self._session

    def get_forecast(self, location: str, days: int = 3) -> dict:
        """Fetch weather data for a location. Returns dict with temp, humidity, rain chance."""
        if not self.api_key:
            logger.warning(
                "OPENWEATHER_API_KEY not set. Returning fallback weather data. "
                "Set OPENWEATHER_API_KEY in .env for real weather data."
            )
            return {
                "location": location,
                "current_temp_c": 28,
                "rain_chance": 85,
                "humidity": 60,
                "note": "Mocked data (OPENWEATHER_API_KEY not configured)",
            }

        try:
            session = self._get_session()
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
            }
            response = session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                "location": data.get("name", location),
                "current_temp_c": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rain_chance": 100 if "rain" in data else 0,
                "description": data["weather"][0]["description"] if data.get("weather") else "",
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"Weather API HTTP error: {e.response.status_code} for {location}")
            return {"error": f"HTTP {e.response.status_code}", "note": f"Failed to fetch weather for {location}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed for {location}: {e}")
            return {"error": str(e), "note": "Failed to fetch real data"}
        except Exception as e:
            logger.error(f"Weather client unexpected error: {e}")
            return {"error": str(e), "note": "Unexpected error"}

    def close(self):
        """Close the underlying HTTP session."""
        if self._session:
            self._session.close()
            self._session = None