import sys
import os
import asyncio
import unittest
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.router.router import IntentRouter
from backend.router.registry import AdvisorRegistry

class TestIntentRouter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.router = IntentRouter()
        self.context = {
            "user_profile": {"location": "Assam", "primary_crop": "Rice"},
            "ml_context": "",
            "recent_history": []
        }

    async def test_crop_routing(self):
        # 1. Specialized crop query should route to CropAdvisor
        query = "What crops are recommended for rainy season?"
        result = await self.router.route(query, self.context)
        self.assertEqual(result.advisor_name, "CropAdvisor")
        self.assertTrue(result.metadata.get("routing_decision_time_ms") < 20.0)  # performance target check

    async def test_dairy_routing(self):
        # 2. Specialized dairy query should route to DairyAdvisor
        query = "How can I increase the fat content in my milk production?"
        result = await self.router.route(query, self.context)
        self.assertEqual(result.advisor_name, "DairyAdvisor")

    async def test_disease_routing(self):
        # 3. Specialized pest/disease query should route to DiseaseAdvisor
        query = "My tomato leaves have yellow spots and blight."
        result = await self.router.route(query, self.context)
        self.assertEqual(result.advisor_name, "DiseaseAdvisor")

    async def test_fallback_routing(self):
        # 4. Out of domain query should fall back to GeneralAdvisor
        query = "Hello, how are you today?"
        result = await self.router.route(query, self.context)
        self.assertEqual(result.advisor_name, "GeneralAdvisor")

if __name__ == "__main__":
    unittest.main()
