import json
import os
import threading
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MemoryManager:
    """Manages short-term conversation memory and persistent farm profiles safely across threads."""
    def __init__(self, storage_file="memory.json"):
        self.lock = threading.Lock()
        self.storage_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.storage_file = os.path.join(self.storage_dir, storage_file)
        self.sessions = {}
        self.profiles = {}
        self._load()

    def _load(self):
        with self.lock:
            if os.path.exists(self.storage_file):
                try:
                    with open(self.storage_file, "r") as f:
                        data = json.load(f)
                        self.sessions = data.get("sessions", {})
                        self.profiles = data.get("profiles", {})
                    logger.info(f"Loaded {len(self.sessions)} sessions and {len(self.profiles)} profiles.")
                except Exception as e:
                    logger.error(f"Failed to load memory: {e}")

    def _save(self):
        with self.lock:
            try:
                with open(self.storage_file, "w") as f:
                    json.dump({"sessions": self.sessions, "profiles": self.profiles}, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save memory: {e}")

    def get_chat_history(self, session_id, max_turns=5):
        """Get recent chat history, formatted for Gemini."""
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            
            # Keep only the last N turns (1 turn = user + assistant usually)
            history = self.sessions[session_id][-max_turns * 2:] if max_turns else self.sessions[session_id]
            return list(history)

    def add_message(self, session_id, role, content):
        """Role should be 'user' or 'model' for Gemini compatibility."""
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append({"role": role, "content": content})
        self._save()

    def set_user_profile(self, user_id, profile_dict):
        with self.lock:
            if user_id not in self.profiles:
                self.profiles[user_id] = {}
            self.profiles[user_id].update(profile_dict)
        self._save()

    def get_user_profile(self, user_id):
        with self.lock:
            return dict(self.profiles.get(user_id, {}))

