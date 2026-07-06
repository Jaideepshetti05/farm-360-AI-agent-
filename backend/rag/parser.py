import os

class DocumentParser:
    @staticmethod
    def parse_file(file_path: str) -> str:
        """
        Parses TXT, Markdown, and configuration files directly.
        Returns text fallbacks for binary documents.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md", ".html", ".jinja2"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            return f"Parsed PDF document content from {os.path.basename(file_path)}."
        elif ext == ".docx":
            return f"Parsed DOCX document content from {os.path.basename(file_path)}."
            
        return f"Raw content of {os.path.basename(file_path)}."
