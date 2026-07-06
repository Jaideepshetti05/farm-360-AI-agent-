import sys
import os
import asyncio
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from backend.core.database import engine, async_session
from backend.models.database import Base
from backend.services.database_service import UnitOfWork
from backend.services.context_builder import PromptContextService
from backend.services.health_service import HealthService
from backend.core.security import encryptor

class TestPhase1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create all tables in the temporary database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        # Drop all tables after test
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def test_db_transaction_and_uow(self):
        # 1. Create a session and add message inside Unit of Work
        async with UnitOfWork() as uow:
            # Create user placeholder
            from backend.models.database import User
            user = User(id="test_farmer", email="test@farm360.com", hashed_password="pw")
            uow.session.add(user)
            await uow.session.flush()

            # Create session
            session = await uow.session_repo.create_session(user_id="test_farmer", title="Test Session")
            session_id = session.id
            
            # Add message
            await uow.session_repo.add_history_message(session_id, "user", "Hello Farm360")
            await uow.session_repo.add_history_message(session_id, "assistant", "Hello farmer!")

        # 2. Re-open transaction and verify history
        async with UnitOfWork() as uow:
            db_session = await uow.session_repo.get_session_by_id(session_id)
            self.assertIsNotNone(db_session)
            self.assertEqual(len(db_session.history), 2)
            self.assertEqual(db_session.history[0].role, "user")
            self.assertEqual(db_session.history[0].content, "Hello Farm360")

    async def test_pii_encryption(self):
        # Verify encryption and decryption transparent wrapper works
        email_plain = "secret_farmer@india.com"
        coords_plain = "28.6139, 77.2090"
        
        async with UnitOfWork() as uow:
            from backend.models.database import User, UserProfile
            user = User(id="farmer_pii", email=email_plain, hashed_password="pw")
            uow.session.add(user)
            await uow.session.flush()
            
            profile = await uow.profile_repo.create_or_update_profile(
                user_id="farmer_pii",
                location="Delhi",
                gps_coordinates=coords_plain
            )
            
        # Re-query and verify decryption works automatically
        async with UnitOfWork() as uow:
            from backend.models.database import User, UserProfile
            from sqlalchemy import select
            
            # Query User
            res = await uow.session.execute(select(User).where(User.id == "farmer_pii"))
            db_user = res.scalars().first()
            self.assertEqual(db_user.email, email_plain)
            
            # Query Profile
            res_p = await uow.session.execute(select(UserProfile).where(UserProfile.user_id == "farmer_pii"))
            db_profile = res_p.scalars().first()
            self.assertEqual(db_profile.gps_coordinates, coords_plain)
            
            # Verify database stores encrypted text, not plain text
            from sqlalchemy import text
            raw_res = await uow.session.execute(text("SELECT email FROM users WHERE id = 'farmer_pii'"))
            raw_email = raw_res.scalars().first()
            self.assertNotEqual(raw_email, email_plain)

    async def test_context_builder_budget(self):
        # Verify prompt builder budget enforcement
        system_template = "System template: {profile} and {ml_context}"
        profile = {"location": "Punjab"}
        ml_context = "Crop yields are high."
        
        history = [
            {"role": "user", "content": "Message A"},
            {"role": "assistant", "content": "Response A"},
            {"role": "user", "content": "Message B"},
            {"role": "assistant", "content": "Response B"},
        ]
        
        # Build prompt under strict budget of 50 tokens (approx 200 chars)
        prompt = PromptContextService.build_prompt_context(
            system_prompt_template=system_template,
            user_profile=profile,
            ml_context=ml_context,
            recent_history=history,
            summary_text="Summarized old messages.",
            max_context_tokens=50
        )
        
        self.assertIsNotNone(prompt)
        self.assertTrue(len(prompt) >= 1)
        self.assertEqual(prompt[0]["role"], "system")

    async def test_health_check_service(self):
        # Verify Health check executes successfully
        health = await HealthService.check_health()
        self.assertIn("status", health)
        self.assertIn("postgres", health)
        self.assertIn("redis", health)

if __name__ == "__main__":
    unittest.main()
