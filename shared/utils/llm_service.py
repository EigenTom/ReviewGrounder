"""
Abstract base class for LLM services
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # "system", "user", "assistant"
    content: str


class LLMService(ABC):
    """Abstract base class for LLM services"""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.8,
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
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty (0-2)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.8,
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
        pass

