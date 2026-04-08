"""
Factory for creating LLM services from configuration
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from .llm_service import LLMService
from .vllm_service import VLLMService
from .gpt_service import GPTService


class LLMServiceFactory:
    """Factory for creating LLM services from configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize factory with configuration
        
        Args:
            config_file: Path to vLLM service config YAML file
        """
        if config_file is None:
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / "shared" / "configs" / "llm_service_config.yaml"
        
        self.config_file = Path(config_file)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def create_vllm_service(self, service_type: str, **override_params) -> VLLMService:
        """
        Create vLLM service from configuration
        
        Args:
            **override_params: Parameters to override config values
            
        Returns:
            VLLMService instance
        """
        vllm_config = self._config.get(service_type, {})
        
        # Merge config with overrides
        params = {
            "base_url": vllm_config.get("base_url", "http://localhost:8000/v1"),
            "api_key": vllm_config.get("api_key", "dummy-key"),
            "model_name": vllm_config.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            "timeout": vllm_config.get("timeout", 300),
        }
        params.update(override_params)
        
        return VLLMService(**params)
    
    def create_gpt_service(self, **override_params) -> GPTService:
        """
        Create GPT service from configuration
        
        Args:
            **override_params: Parameters to override config values
            
        Returns:
            GPTService instance
        """
        gpt_config = self._config.get("gpt", {})
        
        # if not gpt_config.get("enabled", False):
        #     raise ValueError("GPT service is not enabled in configuration")
        
        # Merge config with overrides
        params = {
            "api_key": gpt_config.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            "model_name": gpt_config.get("model_name", "gpt-4o"),
            "base_url": gpt_config.get("base_url"),
            "timeout": gpt_config.get("timeout", 300),
        }
        params.update(override_params)
        
        return GPTService(**params)
    
    def create_service(self, service_type: str = "vllm", **override_params) -> LLMService:
        """
        Create LLM service by type
        
        Args:
            service_type: Service type ("vllm" or "gpt")
            **override_params: Parameters to override config values
            
        Returns:
            LLMService instance
        """
        if service_type.startswith("vllm"):
            return self.create_vllm_service(service_type, **override_params)
        elif service_type == "gpt":
            return self.create_gpt_service(**override_params)
        else:
            raise ValueError(f"Unknown service type: {service_type}")
    
    def get_llm_assignment(self, component: str) -> str:
        """
        Get LLM service assignment for a component

        Args:
            component: One of: insight_miner, results_analyzer, reviewer,
                       keyword_generator, paper_summarizer, refiner

        Returns:
            Service type ("vllm", "gpt", or other configured backend)
        """
        assignments = self._config.get("llm_assignments", {})
        if component in assignments:
            return assignments[component]

        # Fallbacks for backward compatibility
        if component == "refiner" and "reviewer" in assignments:
            return assignments["reviewer"]
        if component == "insight_miner" and "paper_summarizer" in assignments:
            return assignments["paper_summarizer"]
        if component == "results_analyzer" and "paper_summarizer" in assignments:
            return assignments["paper_summarizer"]

        return assignments.get(component, "vllm")
    
    def create_service_for_component(self, component: str, **override_params) -> LLMService:
        """
        Create LLM service for a specific component based on configuration

        Args:
            component: One of: insight_miner, results_analyzer, reviewer,
                       keyword_generator, paper_summarizer, refiner
            **override_params: Parameters to override config values
            
        Returns:
            LLMService instance
        """
        service_type = self.get_llm_assignment(component)
        return self.create_service(service_type, **override_params)


# Global factory instance
_factory: Optional[LLMServiceFactory] = None


def load_api_key_from_config(config_path: str) -> Optional[str]:
    """
    Load API key from a YAML config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        API key string, or None if not found
        
    Note:
        Returns None (instead of raising) if file doesn't exist or key not found,
        to allow graceful fallback to environment variables.
    """
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('api_key')
    except Exception:
        return None


def get_llm_service_factory(config_file: Optional[str] = None) -> LLMServiceFactory:
    """
    Get or create global LLM service factory
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        LLMServiceFactory instance
    """
    global _factory
    if _factory is None or config_file is not None:
        _factory = LLMServiceFactory(config_file)
    return _factory

