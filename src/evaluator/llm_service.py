"""Lightweight wrapper for an OpenAI-compatible chat completion endpoint.

This service is used by downstream scripts (e.g. deepreview_like.py) to call
Qwen2.5-72B or other models exposed via an ngrok forwarded base URL, or
directly call OpenAI GPT models via their API.

Modes:
- "ngrok": Use ngrok-forwarded endpoint (requires ngrok_url)
- "openai": Use OpenAI API directly (uses https://api.openai.com)
"""

from typing import List, Dict, Any, Optional, Literal
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import re

# Optional yaml import for config loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class LLMService:
    def __init__(
        self,
        api_key: str,
        ngrok_url: Optional[str] = None,
        mode: Optional[Literal["ngrok", "openai"]] = None,
    ):
        """
        Initialize LLMService.
        
        Args:
            api_key: API key for authentication
            ngrok_url: URL for ngrok-forwarded endpoint. Required if mode is "ngrok" or not specified.
                      Ignored if mode is "openai". For backward compatibility, can be provided as 
                      second positional argument.
            mode: Service mode - "ngrok" for ngrok-forwarded endpoints, "openai" for OpenAI API.
                  If not specified, defaults to "ngrok" if ngrok_url is provided, otherwise defaults to "openai".
        
        Raises:
            ValueError: If mode is "ngrok" but ngrok_url is not provided, or if invalid mode is specified
        
        Examples:
            # Ngrok mode (backward compatible)
            service = LLMService(api_key="key", ngrok_url="https://...")
            
            # OpenAI mode
            service = LLMService(api_key="key", mode="openai")
            
            # Explicit ngrok mode
            service = LLMService(api_key="key", ngrok_url="https://...", mode="ngrok")
        """
        self.api_key = api_key
        
        # Determine mode if not explicitly provided
        if mode is None:
            if ngrok_url is not None:
                mode = "ngrok"
            else:
                mode = "openai"
        
        self.mode = mode
        
        if mode == "ngrok":
            if ngrok_url is None:
                raise ValueError("ngrok_url is required when mode is 'ngrok'")
            self.base_url = ngrok_url.rstrip("/")
        elif mode == "openai":
            self.base_url = "https://api.openai.com"
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'ngrok' or 'openai'")
        
        # Create a session with retry strategy for better connection handling
        self.session = requests.Session()
        
        # Configure retry strategy - handles SSL errors and connection issues
        # Retry on connection errors (including SSL EOF errors), read timeouts, and server errors
        retry_strategy = Retry(
            total=5,  # Maximum total number of retries
            connect=5,  # Number of retries for connection errors (including SSL)
            read=3,  # Number of retries for read timeouts
            backoff_factor=2,  # Exponential backoff: 2, 4, 8, 16, 32 seconds
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
            allowed_methods=["POST"],  # Only retry POST requests
            respect_retry_after_header=True,  # Respect Retry-After header if present
        )
        
        # Mount adapter with retry strategy and connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools to cache
            pool_maxsize=20,  # Maximum number of connections to save in the pool
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Use session for better connection pooling and retry handling
        # The retry strategy will automatically retry on SSL errors and connection issues
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,  # Increased timeout for long-running requests
                verify=True,  # SSL verification (set to False only if needed for self-signed certs)
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            # These should be retried automatically by the adapter, but log for debugging
            raise requests.exceptions.RequestException(
                f"SSL/Connection error after retries: {e}. "
                f"URL: {self.base_url}/v1/chat/completions"
            ) from e

    def generate_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        result = self._post_chat(payload)
        return result["choices"][0]["message"]["content"].strip()

    def generate_text(self, prompt: str, model: str = "Qwen/Qwen2.5-72B-Instruct", **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(model=model, messages=messages, **kwargs)

# Replace double quotes within text entries with backticks to avoid JSON decode errors
# Match "key": "value" patterns and replace internal quotes in values
def fix_response(response: str) -> str:
    # 1. remove all newline symbols and fix all escape symbols in the response string
    response = response.replace('\\n', ' ').replace('\\"', "'").replace('\\', '\\\\')
    
    # 2. use regex to find all "key": "value" patterns and replace internal quotes in values
    TEXT_KEYS = {"summary", "strengths", "weaknesses", "questions", "decision"}

    # Regex that works line-by-line:
    # ^\s*"key": "value"(,)$
    # where `value` is everything between the first and LAST quote on that line
    pattern = re.compile(
        r'^(\s*"(?P<key>[^"]+)":\s*")(?P<value>.*)("(?P<trail>,?)\s*)$',
        re.MULTILINE
    )

    def replace_quotes_in_value(match):
        key = match.group("key")
        prefix = match.group(1)     # leading spaces + "key": "
        value = match.group("value")
        closing_and_trail = match.group(4)  # final quote + optional comma + spaces
        
        # guardrail for stupid Qwen 2.5 behavior
        if key in TEXT_KEYS: 
            value = value.replace('"', '`')
        
        # fix edge case: review scorings are returned as numbers, not strings
        if type(value) is not str:
            value = str(value)
        return prefix + value + closing_and_trail
    
    fixed_response = pattern.sub(replace_quotes_in_value, response)
    return fixed_response


def parse_llm_response(response: str) -> dict:
    # input str is like ```json....```
    # strip the ```json ... ``` if present
    # match everything within ```json and ```
    try:
        regex_pattern = r"```json(.*?)```"
    
        match = re.search(regex_pattern, response, re.DOTALL)
        if match:
            response = match.group(1).strip()
            
        # fix response to avoid JSON decode errors
        response = fix_response(response)
        
        try:
            res = json.loads(response)
            return res
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON decode error: {e}. Response was:\n{response}")
            return {}
    except Exception as e:
        print(f"[WARN] Unexpected error (possibly stupid format error) during parsing: {e}. Response was:\n{response}")
        return {}
    
def load_api_key_from_config(config_path: str) -> str:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['api_key']

if __name__ == "__main__":
    cfg_path = "configs.yaml"
    api_key = load_api_key_from_config(cfg_path)
    service = LLMService(api_key=api_key, mode="openai")
    demo = service.generate_text(
        "Hello, how are you?", 
        model="gpt-5", 
    )
    print("OpenAI mode demo response:", demo)
    
    
