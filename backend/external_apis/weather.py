import os
import httpx
from loguru import logger


class WeatherClient:
    """OpenWeather API integration using async HTTP client (httpx.AsyncClient)."""
    def __init__(self):
        self.api_key = os.environ.get("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def get_forecast(self, location: str, days: int = 3) -> dict:
        """Fetch weather data for a location asynchronously. Returns dict with temp, humidity, rain chance."""
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
            client = self._get_client()
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
            }
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                "location": data.get("name", location),
                "current_temp_c": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rain_chance": 100 if "rain" in data else 0,
                "description": data["weather"][0]["description"] if data.get("weather") else "",
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Weather API HTTP error: {e.response.status_code} for {location}")
            return {"error": f"HTTP {e.response.status_code}", "note": f"Failed to fetch weather for {location}"}
        except httpx.RequestError as e:
            logger.error(f"Weather API request failed for {location}: {e}")
            return {"error": str(e), "note": "Failed to fetch real data"}
        except Exception as e:
            logger.error(f"Weather client unexpected error: {e}")
            return {"error": str(e), "note": "Unexpected error"}

    async def close(self):
        """Close the underlying async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None