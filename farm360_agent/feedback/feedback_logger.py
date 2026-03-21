import datetime

class FeedbackSystem:
    """Logs corrections and user feedback for active learning."""
    def __init__(self, log_path="feedback.log"):
        self.log_path = log_path

    def log_interaction(self, query, response, rating, comments=""):
        """Rating: 1 (positive), 0 (neutral), -1 (negative)"""
        with open(self.log_path, "a") as f:
            f.write(f"[{datetime.datetime.now()}] Rating: {rating} | Q: {query} | R: {response} | Comment: {comments}\n")
