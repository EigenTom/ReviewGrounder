"""
Simplified vLLM service for review system

This service only handles API calls, no load balancing logic.
Load balancing should be handled at the deployment service level (e.g., nginx reverse proxy).
"""
import os
import time
import random
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from threading import Semaphore, Lock as ThreadLock
from openai import OpenAI
from .llm_service import LLMService, ChatMessage


class VLLMService(LLMService):
    """
    Simplified vLLM service wrapper for local LLM deployment
    
    This service connects to a vLLM server endpoint.
    Load balancing should be handled at the deployment level (e.g., nginx, multiple services behind a load balancer).
    
    Features:
    - Simple API calls to a single endpoint
    - Automatic retry with exponential backoff for 500 errors
    - Configurable max concurrent requests (per service instance)
    """
    
    # Class-level semaphore for rate limiting (shared across all instances of this service)
    # Use lazy initialization to avoid pickle issues with multiprocessing
    _request_semaphore: Optional[Semaphore] = None
    _max_concurrent_requests: int = 8  # Default limit
    _semaphore_lock = ThreadLock()  # Thread-safe initialization lock
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        config_file: Optional[str] = None,
        max_concurrent_requests: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize vLLM service
        
        Args:
            base_url: vLLM server base URL (default: from config or http://localhost:8000/v1)
            api_key: API key (overrides config)
            model_name: Model name identifier (overrides config)
            timeout: Request timeout in seconds (overrides config)
            config_file: Path to config file (default: configs/llm_service_config.yaml)
            max_concurrent_requests: Maximum concurrent requests per service instance (default: 8)
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Initial retry delay in seconds (default: 1.0)
            retry_backoff: Retry delay multiplier (default: 2.0)
        """
        # Load config from YAML
        config = self._load_config(config_file)
        vllm_config = config.get("vllm", {})
        
        # Use provided values or fall back to config, then environment variables
        self.base_url = base_url or vllm_config.get("base_url") or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        
        self.model_name = model_name or vllm_config.get("model_name", "Qwen/Qwen3-4B-Instruct-2507")
        self.api_key = api_key or vllm_config.get("api_key", "dummy-key")
        self.timeout = timeout or vllm_config.get("timeout", 300)
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        
        # Rate limiting: Initialize class-level semaphore if not already initialized
        # Use lazy initialization with thread-safe check to avoid pickle issues
        if max_concurrent_requests is not None:
            VLLMService._max_concurrent_requests = max_concurrent_requests
        else:
            # Try to get from config
            config_max_concurrent = vllm_config.get("max_concurrent_requests")
            if config_max_concurrent is not None:
                VLLMService._max_concurrent_requests = config_max_concurrent
        
        # Lazy initialization of semaphore will happen on first use
        # This avoids pickle issues when using multiprocessing/ThreadPoolExecutor
        
        # Store default sampling parameters from config
        self.default_temperature = vllm_config.get("temperature", 0.7)
        self.default_top_p = vllm_config.get("top_p", 0.8)
        self.default_top_k = vllm_config.get("top_k", 20)
        self.default_max_tokens = vllm_config.get("max_tokens", 16384)
        self.default_presence_penalty = vllm_config.get("presence_penalty", 0.0)
        self.default_enable_thinking = vllm_config.get("enable_thinking", False)
        
        # Create OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
    

    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_file: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if config_file is None:
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / "shared" / "configs" / "llm_service_config.yaml"
        
        config_path = Path(config_file)
        if not config_path.exists():
            # Return defaults if config file doesn't exist
            return {
                "vllm": {
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "dummy-key",
                    "model_name": "Qwen/Qwen3-4B-Instruct-2507",
                    "timeout": 300,
                    "max_concurrent_requests": 8,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_tokens": 16384,
                    "presence_penalty": 0.0,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                }
            }
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    @classmethod
    def _ensure_semaphore(cls):
        """Thread-safe lazy initialization of semaphore to avoid pickle issues"""
        if cls._request_semaphore is None:
            with cls._semaphore_lock:
                # Double-check pattern
                if cls._request_semaphore is None:
                    cls._request_semaphore = Semaphore(cls._max_concurrent_requests)
    
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
    
    def generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from messages
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature (uses config default if None)
            top_p: Top-p sampling parameter (uses config default if None)
            top_k: Top-k sampling parameter (uses config default if None)
            max_tokens: Maximum tokens to generate (uses config default if None)
            presence_penalty: Presence penalty (uses config default if None)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        formatted_messages = self._format_messages(messages)
        
        # Use provided values or fall back to config defaults
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        presence_penalty = presence_penalty if presence_penalty is not None else self.default_presence_penalty
        
        # Ensure semaphore is initialized (lazy, thread-safe)
        self._ensure_semaphore()
        
        # Use semaphore to limit concurrent requests
        with VLLMService._request_semaphore:
            last_exception = None
            
            for retry_attempt in range(self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        presence_penalty=presence_penalty,
                        extra_body={"chat_template_kwargs": {"enable_thinking": self.default_enable_thinking}},
                        **kwargs
                    )
                    
                    return response.choices[0].message.content
                    
                    # llama_3 long context ver has such a quirk
                    # return response.choices[0].message.content.replace("</s>", "")
                
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if it's a server error (500, 502, 503, 504) that we should retry
                    should_retry = False
                    error_str = str(e).lower()
                    
                    if any(code in error_str for code in ["500", "502", "503", "504"]):
                        should_retry = True
                    elif "server error" in error_str or "internal server error" in error_str:
                        should_retry = True
                    
                    # Don't retry on last attempt
                    if retry_attempt < self.max_retries and should_retry:
                        # Calculate delay with exponential backoff and jitter
                        delay = self.retry_delay * (self.retry_backoff ** retry_attempt)
                        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                        time.sleep(delay + jitter)
                        continue
                    else:
                        # Either not a retryable error or out of retries
                        raise last_exception
    
    def stream_generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        **kwargs
    ):
        """
        Stream generate text from messages
        
        Yields:
            Generated text chunks
        """
        formatted_messages = self._format_messages(messages)
        
        # Use provided values or fall back to config defaults
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        presence_penalty = presence_penalty if presence_penalty is not None else self.default_presence_penalty
        
        # Ensure semaphore is initialized (lazy, thread-safe)
        self._ensure_semaphore()
        
        # Use semaphore to limit concurrent requests
        with VLLMService._request_semaphore:
            last_exception = None
            
            for retry_attempt in range(self.max_retries + 1):
                try:
                    stream = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        presence_penalty=presence_penalty,
                        stream=True,
                        **kwargs
                    )
                    
                    # Stream chunks
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    
                    return  # Success, exit retry loop
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if it's a server error that we should retry
                    should_retry = False
                    error_str = str(e).lower()
                    
                    if any(code in error_str for code in ["500", "502", "503", "504"]):
                        should_retry = True
                    elif "server error" in error_str or "internal server error" in error_str:
                        should_retry = True
                    
                    # Don't retry on last attempt
                    if retry_attempt < self.max_retries and should_retry:
                        # Calculate delay with exponential backoff and jitter
                        delay = self.retry_delay * (self.retry_backoff ** retry_attempt)
                        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                        time.sleep(delay + jitter)
                        continue
                    else:
                        # Either not a retryable error or out of retries
                        raise last_exception
