import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MemoryManager:
    """Manages short-term conversation memory and persistent farm profiles safely across threads."""
    def __init__(self, storage_file="memory.json"):
        import threading
        self.lock = threading.Lock()
        self.storage_file = os.path.join(BASE_DIR, "logs", storage_file)
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
                except Exception:
                    pass

    def _save(self):
        with self.lock:
            try:
                with open(self.storage_file, "w") as f:
                    json.dump({"sessions": self.sessions, "profiles": self.profiles}, f, indent=4)
            except Exception:
                pass

    def get_chat_history(self, session_id):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            return list(self.sessions[session_id])

    def add_message(self, session_id, role, content):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append({"role": role, "content": content})
        self._save()

    def set_user_profile(self, user_id, profile_dict):
        """E.g., {"location": "Assam", "farm_size": 5, "primary_crop": "Rice"}"""
        with self.lock:
            self.profiles[user_id] = profile_dict
        self._save()

    def get_user_profile(self, user_id):
        with self.lock:
            return dict(self.profiles.get(user_id, {}))
