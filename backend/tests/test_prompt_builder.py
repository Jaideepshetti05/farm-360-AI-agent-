import sys
import os
import asyncio
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from backend.core.database import engine
from backend.models.database import Base
from backend.services.database_service import UnitOfWork
from backend.services.prompt_service import PromptService
from backend.models.database import PromptTemplate

class TestPromptBuilder(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Build clean database schemas
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        # Clean up database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def test_static_template_loading(self):
        # 1. Verify static prompt templates load successfully from filesystem
        text, config = PromptService.get_template("general_assistant", "1.0.0")
        self.assertIsNotNone(text)
        self.assertIn("FARMER PROFILE", text)
        self.assertIn("RESPONSE RULES", text)

    def test_prompt_rendering_and_validation(self):
        # 2. Test successful placeholders rendering
        variables = {
            "profile": "Farmer John",
            "ml_context": "High humidity forecast."
        }
        rendered, config = PromptService.render_and_validate("general_assistant", variables)
        self.assertIn("Farmer John", rendered)
        self.assertIn("High humidity forecast.", rendered)
        
        # 3. Test missing placeholder exception
        bad_variables = {
            "profile": "Farmer John"
            # ml_context is missing!
        }
        with self.assertRaises(ValueError):
            PromptService.render_and_validate("general_assistant", bad_variables)

    async def test_database_prompt_overriding(self):
        # 4. Verify that templates stored in DB override disk templates
        db_prompt_text = "Custom database override prompt: {{ profile }} and {{ ml_context }}"
        
        async with UnitOfWork() as uow:
            await uow.prompt_repo.save_or_update(
                name="general_assistant",
                version="1.0.0",
                template_text=db_prompt_text,
                config={"temperature": 0.9}
            )

        # 5. Read prompt template again and assert database override is selected
        rendered, config = PromptService.render_and_validate(
            "general_assistant", 
            {"profile": "Farmer Bob", "ml_context": "None"},
            version="1.0.0"
        )
        self.assertIn("Custom database override prompt", rendered)
        self.assertEqual(config.get("temperature"), 0.9)

if __name__ == "__main__":
    unittest.main()
