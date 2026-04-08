"""
OpenAI GPT API service implementation
"""
import os
from typing import List, Dict, Optional, Any, Union
from openai import OpenAI
from .llm_service import LLMService, ChatMessage


class GPTService(LLMService):
    """
    OpenAI GPT API service wrapper
    
    This service connects to OpenAI's API (or compatible API)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        base_url: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize GPT service
        
        Args:
            api_key: OpenAI / OpenRouter API key (set via env if omitted)
            model_name: Model name (e.g., gpt-4o, openai/gpt-oss-120b:free, etc.)
            base_url: API base URL (default: https://api.openai.com/v1)
            timeout: Request timeout in seconds
        """
        # Prefer explicit parameter, then common environment variables.
        # This allows using OpenRouter (OPENROUTER_API_KEY) without hard-coding secrets.
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
                "environment variable, or pass api_key parameter."
            )
        
        self.model_name = model_name
        self._use_responses_api = "gpt-5.2" in (model_name or "").lower()
        # Prefer explicit base_url, then environment variables, then OpenAI default.
        # This allows swapping in any OpenAI-compatible endpoint (e.g., OpenRouter)
        # without changing code.
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.timeout = timeout
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
    
    def _format_messages(self, messages: List[Union[ChatMessage, Dict[str, str]]]) -> List[Dict[str, str]]:
        """Format messages for OpenAI API"""
        formatted = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        return formatted

    def _responses_output_text(self, response: Any) -> str:
        """Extract text from OpenAI Responses API response."""
        if getattr(response, "output_text", None):
            return response.output_text or ""
        if getattr(response, "output", None) and response.output:
            for item in response.output:
                content = getattr(item, "content", None) or []
                for block in content:
                    if getattr(block, "text", None):
                        return block.text
        return ""

    def generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 16384,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from messages
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter (not used by GPT API, but kept for compatibility)
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty (0-2)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        formatted_messages = self._format_messages(messages)
        
        try:
            if self._use_responses_api:
                # OpenAI Responses API for gpt-5.2 with reasoning effort "none"
                # https://platform.openai.com/docs/api-reference/responses/create
                input_list = [
                    {"role": m["role"], "content": m["content"]}
                    for m in formatted_messages
                ]
                response = self.client.responses.create(
                    model=self.model_name,
                    input=input_list,
                    reasoning={"effort": "none"},
                    max_output_tokens=max_tokens,
                )
                return self._responses_output_text(response)
            # GPT API doesn't support top_k, so we exclude it
            # Some newer models (like GPT 5.2) use max_completion_tokens instead of max_tokens
            params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
            }
            
            # Check if model requires max_completion_tokens instead of max_tokens
            # Models that use max_completion_tokens: o1, o1-preview, o1-mini, and newer models
            if any(model_name in self.model_name.lower() for model_name in ["o1", "gpt-5", "gpt5"]):
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
            
            params.update({k: v for k, v in kwargs.items() if k not in ["top_k", "max_tokens", "max_completion_tokens"]})
            
            response = self.client.chat.completions.create(**params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            # If max_tokens fails, try max_completion_tokens as fallback
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                try:
                    params = {
                        "model": self.model_name,
                        "messages": formatted_messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_completion_tokens": max_tokens,
                        "presence_penalty": presence_penalty,
                    }
                    params.update({k: v for k, v in kwargs.items() if k not in ["top_k", "max_tokens", "max_completion_tokens"]})
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content
                except Exception as e2:
                    raise RuntimeError(f"Error generating text from GPT service: {e2}")
            raise RuntimeError(f"Error generating text from GPT service: {e}")
    
    def stream_generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 16384,
        presence_penalty: float = 0.0,
        **kwargs
    ):
        """
        Stream generate text from messages
        
        Yields:
            Generated text chunks
        """
        formatted_messages = self._format_messages(messages)
        
        try:
            params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "stream": True,
            }
            
            # Check if model requires max_completion_tokens instead of max_tokens
            if any(model_name in self.model_name.lower() for model_name in ["o1", "gpt-5", "gpt5"]):
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
            
            params.update({k: v for k, v in kwargs.items() if k not in ["top_k", "max_tokens", "max_completion_tokens"]})
            
            stream = self.client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            # If max_tokens fails, try max_completion_tokens as fallback
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                try:
                    params = {
                        "model": self.model_name,
                        "messages": formatted_messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_completion_tokens": max_tokens,
                        "presence_penalty": presence_penalty,
                        "stream": True,
                    }
                    params.update({k: v for k, v in kwargs.items() if k not in ["top_k", "max_tokens", "max_completion_tokens"]})
                    stream = self.client.chat.completions.create(**params)
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e2:
                    raise RuntimeError(f"Error streaming text from GPT service: {e2}")
            raise RuntimeError(f"Error streaming text from GPT service: {e}")

