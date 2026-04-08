"""
VLLM Endpoint Pool Manager

Manage multiple vLLM endpoints, implement round-robin access and load balancing.
"""
import random
from pathlib import Path
from typing import List, Optional, Dict
from threading import Lock


class VLLMEndpointPool:
    """
    VLLM Endpoint Pool Manager
    
    Features:
    1. Load multiple vLLM endpoints from file
    2. Round-robin access endpoints (ensure uniform distribution)
    3. Track usage status and errors for each endpoint
    4. Smart selection (based on error rate and success rate, as backup)
    """
    
    def __init__(self, pool_path: Optional[str] = None, endpoints: Optional[List[str]] = None, use_round_robin: bool = True):
        """
        Initialize VLLM Endpoint Pool
        
        Args:
            pool_path: Endpoints file path (one endpoint URL per line)
            endpoints: directly provide endpoints list (prior to pool_path)
            use_round_robin: whether to use round-robin strategy (True=uniform distribution, False=smart selection)
        """
        self.endpoints: List[str] = []
        self.current_index: int = 0  # Round-robin current index
        self.used_indices: List[int] = []  # used indices in current round (for smart selection)
        self.endpoint_status: Dict[str, Dict] = {}  # status information for each endpoint
        self.lock = Lock()  # thread safe lock
        self.use_round_robin = use_round_robin  # whether to use round-robin
        
        # load endpoints
        if endpoints:
            self.endpoints = [e.strip() for e in endpoints if e.strip()]
        elif pool_path:
            self._load_from_file(pool_path)
        else:
            # try to get single endpoint from environment variable (backward compatibility)
            import os
            env_endpoint = os.environ.get("VLLM_BASE_URL")
            if env_endpoint:
                # ensure format is correct (may need to add /v1)
                if not env_endpoint.endswith('/v1'):
                    if env_endpoint.endswith('/'):
                        env_endpoint = env_endpoint.rstrip('/') + '/v1'
                    else:
                        env_endpoint = env_endpoint + '/v1'
                self.endpoints = [env_endpoint]
        
        if not self.endpoints:
            raise ValueError(
                "No vLLM endpoints available. Provide endpoints via pool_path, endpoints parameter, "
                "or VLLM_BASE_URL environment variable."
            )
        
        # initialize status for each endpoint
        for endpoint in self.endpoints:
            self.endpoint_status[endpoint] = {
                'error_count': 0,
                'last_error_time': None,
                'consecutive_errors': 0,
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0.0,  # 累计响应时间
            }
    
    def _load_from_file(self, pool_path: str):
        """load vLLM endpoints from file"""
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
                f"VLLM endpoint pool file not found: {pool_path} (tried: {path})"
            )
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        endpoints = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # ensure format is correct (may need to add /v1)
                if not line.endswith('/v1'):
                    if line.endswith('/'):
                        line = line.rstrip('/') + '/v1'
                    else:
                        line = line + '/v1'
                endpoints.append(line)
        
        self.endpoints = endpoints
        
        if not self.endpoints:
            raise ValueError(f"No valid vLLM endpoints found in pool file: {pool_path}")
    
    def get_endpoint(self) -> str:
        """
        Get next available endpoint (round-robin strategy)
        
        Strategy:
        - Round-robin mode (default): simple round-robin, ensure uniform distribution
        - Smart selection mode: select best endpoint based on error rate, success rate, response time
        
        Returns:
            Available endpoint URL
        """
        import time
        
        with self.lock:
            if not self.endpoints:
                raise ValueError("No vLLM endpoints available in pool")
            
            if self.use_round_robin:
                # Round-robin: simple round-robin, ensure uniform distribution
                selected_idx = self.current_index
                self.current_index = (self.current_index + 1) % len(self.endpoints)
                
                selected_endpoint = self.endpoints[selected_idx]
                self.endpoint_status[selected_endpoint]['total_requests'] += 1
                
                return selected_endpoint
            else:
                # smart selection mode (original logic)
                # if current round is complete, start a new round
                if len(self.used_indices) >= len(self.endpoints):
                    self.used_indices = []
                
                # get indices not used in current round
                available_indices = [i for i in range(len(self.endpoints)) if i not in self.used_indices]
                
                if not available_indices:
                    # all endpoints are in current round, start a new round
                    available_indices = list(range(len(self.endpoints)))
                    self.used_indices = []
                
                # prioritize endpoints with fewer errors and higher success rate
                endpoint_scores = []
                for idx in available_indices:
                    endpoint = self.endpoints[idx]
                    status = self.endpoint_status[endpoint]
                    
                    # calculate score: error count, success rate, response time, score越高
                    error_count = status['error_count']
                    total = status['total_requests']
                    success_rate = (status['successful_requests'] / total) if total > 0 else 1.0
                    
                    # calculate average response time (shorter is better)
                    avg_response_time = (
                        status['total_response_time'] / status['successful_requests']
                        if status['successful_requests'] > 0 else 0.0
                    )
                    # normalize response time score (assume 10 seconds as baseline, faster score higher)
                    response_time_score = 1.0 / (1.0 + avg_response_time / 10.0)
                    
                    # if recent error, reduce score
                    recent_error_penalty = 0
                    if status['last_error_time']:
                        time_since_error = time.time() - status['last_error_time']
                        if time_since_error < 60:  # 1 minute内
                            recent_error_penalty = 0.5
                    
                    score = success_rate - (error_count * 0.1) - recent_error_penalty + (response_time_score * 0.2)
                    endpoint_scores.append((idx, score))
                
                # sort by score, select highest score (but add some randomness)
                endpoint_scores.sort(key=lambda x: x[1], reverse=True)
                
                # select from top 50% (add randomness but prioritize better)
                top_n = max(1, len(endpoint_scores) // 2) if len(endpoint_scores) > 1 else 1
                selected_idx, _ = random.choice(endpoint_scores[:top_n])
                
                # mark as used
                self.used_indices.append(selected_idx)
                
                selected_endpoint = self.endpoints[selected_idx]
                self.endpoint_status[selected_endpoint]['total_requests'] += 1
                
                return selected_endpoint
    
    def mark_success(self, endpoint: str, response_time: float = 0.0):
        """
        mark endpoint as successful
        
        Args:
            endpoint: successful endpoint URL
            response_time: response time (seconds)
        """
        with self.lock:
            if endpoint in self.endpoint_status:
                status = self.endpoint_status[endpoint]
                status['successful_requests'] += 1
                status['consecutive_errors'] = 0
                status['total_response_time'] += response_time
    
    def mark_error(self, endpoint: str, error_type: str = "server_error"):
        """
        mark endpoint as failed
        
        Args:
            endpoint: failed endpoint URL
            error_type: error type ("server_error", "timeout", "connection_error", "other")
        """
        import time
        
        with self.lock:
            if endpoint in self.endpoint_status:
                status = self.endpoint_status[endpoint]
                status['error_count'] += 1
                status['consecutive_errors'] += 1
                status['last_error_time'] = time.time()
    
    def get_status(self) -> Dict:
        """get pool status information (for debugging)"""
        with self.lock:
            return {
                'total_endpoints': len(self.endpoints),
                'current_round_progress': f"{len(self.used_indices)}/{len(self.endpoints)}",
                'endpoints_status': {
                    endpoint: {
                        'error_count': status['error_count'],
                        'successful_requests': status['successful_requests'],
                        'total_requests': status['total_requests'],
                        'success_rate': (
                            status['successful_requests'] / status['total_requests']
                            if status['total_requests'] > 0 else 0.0
                        ),
                        'avg_response_time': (
                            status['total_response_time'] / status['successful_requests']
                            if status['successful_requests'] > 0 else 0.0
                        ),
                        'consecutive_errors': status['consecutive_errors'],
                        'last_error_time': status['last_error_time'],
                    }
                    for endpoint, status in self.endpoint_status.items()
                }
            }
    
    def reset_round(self):
        """reset current round (force start a new round)"""
        with self.lock:
            self.used_indices = []
