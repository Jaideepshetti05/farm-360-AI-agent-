import json
import os
import threading
import tempfile
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Max number of sessions stored in memory to prevent unbounded growth
MAX_SESSIONS = 100
# Max messages per session to cap per-user growth
MAX_MESSAGES_PER_SESSION = 200


class MemoryManager:
    """Manages short-term conversation memory and persistent farm profiles safely across threads."""
    def __init__(self, storage_file="memory.json"):
        self.lock = threading.Lock()
        self.storage_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.storage_file = os.path.join(self.storage_dir, storage_file)
        self.sessions = {}
        self.profiles = {}
        self._dirty = False  # Track if we need to save
        self._load()

    def _load(self):
        with self.lock:
            if os.path.exists(self.storage_file):
                try:
                    with open(self.storage_file, "r") as f:
                        data = json.load(f)
                        self.sessions = data.get("sessions", {})
                        self.profiles = data.get("profiles", {})

                    # Enforce caps on load
                    self._enforce_caps()
                    logger.info(
                        f"Loaded {len(self.sessions)} sessions and {len(self.profiles)} profiles."
                    )
                except Exception as e:
                    logger.error(f"Failed to load memory: {e}")

    def _enforce_caps(self):
        """Enforce maximum session count and per-session message limits."""
        # Trim oldest sessions if over cap
        if len(self.sessions) > MAX_SESSIONS:
            excess = sorted(self.sessions.keys())[: len(self.sessions) - MAX_SESSIONS]
            for k in excess:
                del self.sessions[k]
            logger.warning(f"Trimmed {len(excess)} excess sessions (max {MAX_SESSIONS})")

        # Trim per-session messages
        for sid in list(self.sessions.keys()):
            if len(self.sessions[sid]) > MAX_MESSAGES_PER_SESSION:
                self.sessions[sid] = self.sessions[sid][-MAX_MESSAGES_PER_SESSION:]
                logger.debug(f"Trimmed session {sid} to {MAX_MESSAGES_PER_SESSION} messages")

    def _save(self):
        """Atomic save: write to temp file, then rename."""
        with self.lock:
            if not self._dirty:
                return  # Skip if nothing changed
            try:
                # Write to temporary file first
                tmp = tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=self.storage_dir,
                    prefix="memory_",
                    suffix=".tmp",
                    delete=False,
                )
                json.dump(
                    {"sessions": self.sessions, "profiles": self.profiles},
                    tmp,
                    indent=4,
                )
                tmp.close()

                # Atomic rename (os.replace is atomic on POSIX and Windows)
                os.replace(tmp.name, self.storage_file)
                self._dirty = False
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
                # Enforce max sessions before creating new one
                if len(self.sessions) >= MAX_SESSIONS:
                    # Remove oldest session
                    oldest = min(self.sessions.keys(), key=lambda k: (
                        self.sessions[k][0]["timestamp"] if self.sessions[k] and isinstance(self.sessions[k][0], dict) and "timestamp" in self.sessions[k][0] else ""
                    ))
                    del self.sessions[oldest]
                    logger.debug(f"Evicted oldest session {oldest} to make room")

                self.sessions[session_id] = []

            self.sessions[session_id].append({"role": role, "content": content})

            # Enforce per-session cap
            if len(self.sessions[session_id]) > MAX_MESSAGES_PER_SESSION:
                self.sessions[session_id] = self.sessions[session_id][-MAX_MESSAGES_PER_SESSION:]

            self._dirty = True

        # Save after every message (consider batching with a timer in production)
        self._save()

    def set_user_profile(self, user_id, profile_dict):
        with self.lock:
            if user_id not in self.profiles:
                self.profiles[user_id] = {}
            self.profiles[user_id].update(profile_dict)
            self._dirty = True
        self._save()

    def get_user_profile(self, user_id):
        with self.lock:
            return dict(self.profiles.get(user_id, {}))

    def clear_session(self, session_id):
        """Clear a specific session's history."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self._dirty = True
        self._save()