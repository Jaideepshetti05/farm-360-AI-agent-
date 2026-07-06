import os
import re
from typing import Optional, Dict, Any, Tuple
from loguru import logger
from jinja2 import Template
from backend.services.database_service import UnitOfWork
from backend.memory.session import run_async_sync

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "prompts", "templates")

class PromptService:
    @staticmethod
    def _load_static_template(name: str, version: str) -> Optional[str]:
        """Loads prompt from the local static template folder."""
        filename = f"{name}_v{version}.jinja2"
        filepath = os.path.join(TEMPLATES_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"[PromptService] Failed to read static template file {filename}: {e}")
        return None

    @classmethod
    def get_template(cls, name: str, version: str = "1.0.0") -> Tuple[str, Dict[str, Any]]:
        """
        Retrieves a template by name and version.
        Checks database registry first, then falls back to static files.
        Returns a tuple: (template_text, config_dict).
        """
        async def _fetch():
            async with UnitOfWork() as uow:
                prompt_db = await uow.prompt_repo.get_by_name(name, version)
                if prompt_db:
                    return prompt_db.template_text, prompt_db.config or {}
            return None
            
        try:
            db_result = run_async_sync(_fetch())
            if db_result:
                return db_result
        except Exception as e:
            logger.warning(f"[PromptService] DB check failed for prompt {name}: {e}. Falling back to disk.")
            
        # Fallback to local file-system template
        static_text = cls._load_static_template(name, version)
        if static_text is not None:
            # Return baseline defaults for configurations
            default_config = {
                "temperature": 0.2,
                "max_tokens": 1500,
                "model_preference": None
            }
            return static_text, default_config
            
        # If version not found, try to fetch version 1.0.0 as safety fallback
        if version != "1.0.0":
            logger.warning(f"[PromptService] Prompt {name} version {version} not found. Attempting 1.0.0 fallback.")
            return cls.get_template(name, "1.0.0")
            
        raise ValueError(f"Prompt template '{name}' (version {version}) could not be resolved from DB or disk.")

    @classmethod
    def render_and_validate(cls, name: str, variables: Dict[str, Any], version: str = "1.0.0") -> Tuple[str, Dict[str, Any]]:
        """
        Loads the template, renders it with variables via Jinja2,
        and validates that no variables are unresolved.
        Returns a tuple: (rendered_prompt, config_dict).
        """
        template_text, config = cls.get_template(name, version)
        
        from jinja2 import Environment, StrictUndefined
        # 1. Render template using Jinja2
        try:
            env = Environment(undefined=StrictUndefined)
            template = env.from_string(template_text)
            rendered = template.render(**variables)
        except Exception as e:
            logger.error(f"[PromptService] Jinja2 rendering error for template '{name}': {e}")
            raise ValueError(f"Rendering failed for template '{name}': {e}")
            
        # 2. Validate for missing variables (unresolved double curly braces)
        missing_placeholders = re.findall(r"\{\{\s*(\w+)\s*\}\}", rendered)
        if missing_placeholders:
            err_msg = f"Prompt validation failed: Unresolved placeholders detected in template '{name}': {list(set(missing_placeholders))}"
            logger.error(f"[PromptService] {err_msg}")
            raise ValueError(err_msg)
            
        # 3. Check for empty output
        if not rendered.strip():
            raise ValueError(f"Prompt validation failed: Rendered template '{name}' is empty.")
            
        return rendered, config
