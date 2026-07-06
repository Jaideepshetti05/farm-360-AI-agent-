from typing import Optional, List, Dict, Any
from loguru import logger
import json

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    logger.warning("[ContextBuilder] tiktoken package not installed. Falling back to char-length estimation.")

class PromptContextService:
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (cl100k_base) or fallback character count approximation."""
        if not text:
            return 0
        if _TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Error encoding with tiktoken: {e}")
        # Standard fallback: ~4 characters per token
        return len(text) // 4

    @classmethod
    def build_prompt_context(
        cls,
        system_prompt_template: str,
        user_profile: Dict[str, Any],
        ml_context: str,
        recent_history: List[Dict[str, Any]],
        summary_text: Optional[str] = None,
        max_context_tokens: int = 4096
    ) -> List[Dict[str, Any]]:
        """
        Assembles System Prompt, User Profile, ML Context, Summaries, and Chat History 
        into a token-budgeted list of OpenAI-compatible message dicts.
        """
        # 1. Format the core system prompt
        if system_prompt_template in ["general_assistant", "crop_advisor", "disease_advisor", "animal_health", "dairy"]:
            from backend.services.prompt_service import PromptService
            variables = {
                "profile": json.dumps(user_profile, ensure_ascii=False),
                "ml_context": ml_context
            }
            formatted_system, config = PromptService.render_and_validate(system_prompt_template, variables)
        else:
            from jinja2 import Template
            try:
                template = Template(system_prompt_template)
                formatted_system = template.render(
                    profile=json.dumps(user_profile, ensure_ascii=False),
                    ml_context=ml_context
                )
            except Exception:
                formatted_system = system_prompt_template.format(
                    profile=json.dumps(user_profile, ensure_ascii=False),
                    ml_context=ml_context
                )
        
        system_tokens = cls.count_tokens(formatted_system)
        
        # Determine remaining budget
        remaining_budget = max_context_tokens - system_tokens
        
        # 2. Add summary if available
        summary_msg = None
        if summary_text:
            formatted_summary = f"[System Summary of older messages]: {summary_text}"
            summary_tokens = cls.count_tokens(formatted_summary)
            if summary_tokens < remaining_budget:
                summary_msg = {"role": "system", "content": formatted_summary}
                remaining_budget -= summary_tokens
                
        # 3. Add recent messages (sliding window, newest first)
        messages_to_add = []
        for msg in reversed(recent_history):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            # OpenAI API values mapping
            role_mapped = "user" if role == "user" else "assistant"
            msg_dict = {"role": role_mapped, "content": content}
            
            msg_tokens = cls.count_tokens(content)
            if msg_tokens < remaining_budget:
                messages_to_add.insert(0, msg_dict)
                remaining_budget -= msg_tokens
            else:
                logger.info("[ContextBuilder] Truncating history at message count limit (budget exceeded).")
                break
                
        # 4. Construct final payload
        final_payload = [{"role": "system", "content": formatted_system}]
        if summary_msg:
            final_payload.append(summary_msg)
        final_payload.extend(messages_to_add)
        
        return final_payload
