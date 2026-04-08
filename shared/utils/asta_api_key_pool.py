"""
Asta API Key Pool Manager

Manage multiple Asta API keys, implement key rotation and error handling.
"""
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Dict
from threading import Lock


class AstaAPIKeyPool:
    """
    Asta API Key Pool Manager
    
    Features:
    1. Load multiple API keys from file
    2. Randomly rotate keys
    3. Track each key's usage status and errors
    4. Implement debounce retry strategy
    """
    
    def __init__(self, pool_path: Optional[str] = None, keys: Optional[List[str]] = None):
        """
        Initialize API Key Pool
        
        Args:
            pool_path: API keys file path (one key per line)
            keys: directly provide keys list (prior to pool_path)
        """
        self.keys: List[str] = []
        self.used_indices: List[int] = []  # indices used in current rotation
        self.key_status: Dict[str, Dict] = {}  # status information for each key
        self.lock = Lock()  # thread safe lock
        
        # load keys
        if keys:
            self.keys = [k.strip() for k in keys if k.strip()]
        elif os.environ.get("ASTA_API_KEY"):
            # Try to get one or more keys from environment variable (comma-separated)
            self.keys = [k.strip() for k in os.environ.get("ASTA_API_KEY").split(",") if k.strip()]
        elif pool_path:
            self._load_from_file(pool_path)
        else:
            raise ValueError(
                "No API keys available. Provide keys via pool_path, keys parameter, "
                "or ASTA_API_KEY environment variable."
            )
            
        if not self.keys:
            raise ValueError(
                "No API keys available. Provide keys via pool_path, keys parameter, "
                "or ASTA_API_KEY environment variable."
            )
        
        # initialize status for each key
        for key in self.keys:
            self.key_status[key] = {
                'error_count': 0,
                'last_error_time': None,
                'consecutive_errors': 0,
                'total_requests': 0,
                'successful_requests': 0,
            }
    
    def _load_from_file(self, pool_path: str):
        """Load API keys from file"""
        path = Path(pool_path)
        
        # if relative path, try to find file relative to shared/configs
        if not path.is_absolute():
            # try to find file relative to project root
            project_root = Path(__file__).parent.parent.parent
            path = project_root / "shared" / "configs" / pool_path
            if not path.exists():
                # try to find file relative to shared/configs
                path = Path(__file__).parent.parent / "configs" / pool_path
        
        if not path.exists():
            raise FileNotFoundError(
                f"API key pool file not found: {pool_path} (tried: {path})"
            )
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.keys = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if not self.keys:
            raise ValueError(f"No valid API keys found in pool file: {pool_path}")
    
    def get_key(self) -> str:
        """
        Get next available API key (rotation strategy)
        
        Strategy:
        1. If current rotation is not complete, continue using unused keys
        2. If current rotation is complete, start a new round (reset used_indices)
        3. Prioritize keys with no recent errors
        
        Returns:
            Available API key
        """
        with self.lock:
            if not self.keys:
                raise ValueError("No API keys available in pool")
            
            # if current rotation is complete, start a new round
            if len(self.used_indices) >= len(self.keys):
                self.used_indices = []
            
            # get indices not used in current rotation
            available_indices = [i for i in range(len(self.keys)) if i not in self.used_indices]
            
            if not available_indices:
                # all keys are used in current rotation, start a new round
                available_indices = list(range(len(self.keys)))
                self.used_indices = []
            
            # prioritize keys with fewer errors (randomly select, but prioritize keys with higher success rate and fewer errors)
            key_scores = []
            for idx in available_indices:
                key = self.keys[idx]
                status = self.key_status[key]
                
                # calculate score: error count, success rate, score越高
                error_count = status['error_count']
                total = status['total_requests']
                success_rate = (status['successful_requests'] / total) if total > 0 else 1.0
                
                # if recent error, reduce score
                recent_error_penalty = 0
                if status['last_error_time']:
                    time_since_error = time.time() - status['last_error_time']
                    if time_since_error < 60:  # 1 minute
                        recent_error_penalty = 0.5
                
                score = success_rate - (error_count * 0.1) - recent_error_penalty
                key_scores.append((idx, score))
            
            # sort by score, select highest score (but add some randomness)
            key_scores.sort(key=lambda x: x[1], reverse=True)
            
            # select from top 50% (add randomness but prioritize better keys)
            top_n = max(1, len(key_scores) // 2) if len(key_scores) > 1 else 1
            selected_idx, _ = random.choice(key_scores[:top_n])
            
            # mark as used
            self.used_indices.append(selected_idx)
            
            selected_key = self.keys[selected_idx]
            self.key_status[selected_key]['total_requests'] += 1
            
            return selected_key
    
    def mark_success(self, key: str):
        """mark key as successful"""
        with self.lock:
            if key in self.key_status:
                self.key_status[key]['successful_requests'] += 1
                self.key_status[key]['consecutive_errors'] = 0
    
    def mark_error(self, key: str, error_type: str = "rate_limit"):
        """
        mark key as failed
        
        Args:
            key: failed API key
            error_type: error type ("rate_limit", "auth_error", "server_error", "other")
        """
        with self.lock:
            if key in self.key_status:
                status = self.key_status[key]
                status['error_count'] += 1
                status['consecutive_errors'] += 1
                status['last_error_time'] = time.time()
    
    def get_status(self) -> Dict:
        """get pool status information (for debugging)"""
        with self.lock:
            return {
                'total_keys': len(self.keys),
                'current_round_progress': f"{len(self.used_indices)}/{len(self.keys)}",
                'keys_status': {
                    key: {
                        'error_count': status['error_count'],
                        'successful_requests': status['successful_requests'],
                        'total_requests': status['total_requests'],
                        'success_rate': (
                            status['successful_requests'] / status['total_requests']
                            if status['total_requests'] > 0 else 0.0
                        ),
                        'consecutive_errors': status['consecutive_errors'],
                        'last_error_time': status['last_error_time'],
                    }
                    for key, status in self.key_status.items()
                }
            }
    
    def reset_round(self):
        """reset current rotation (force start a new round)"""
        with self.lock:
            self.used_indices = []
