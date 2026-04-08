"""
Reranker Endpoint Pool Manager

Manage multiple reranker API endpoints, implement round-robin access and load balancing.
Reuse the logic of VLLMEndpointPool.
"""
from pathlib import Path
from typing import List, Optional, Dict
from threading import Lock


class RerankerEndpointPool:
    """
    Reranker Endpoint Pool Manager
    
    Features:
    1. Load multiple reranker API endpoints from file
    2. Round-robin access endpoints (ensure uniform distribution)
    3. Track usage status and errors for each endpoint
    """
    
    def __init__(self, pool_path: Optional[str] = None, endpoints: Optional[List[str]] = None, use_round_robin: bool = True):
        """
        Initialize Reranker Endpoint Pool
        
        Args:
            pool_path: Endpoints file path (one endpoint URL per line)
            endpoints: directly provide endpoints list (prior to pool_path)
            use_round_robin: whether to use round-robin strategy (default True, recommended)
        """
        self.endpoints: List[str] = []
        self.current_index: int = 0  # Round-robin current index
        self.endpoint_status: Dict[str, Dict] = {}  # status information for each endpoint
        self.lock = Lock()  # thread safe lock
        self.use_round_robin = use_round_robin  # whether to use round-robin
        
        # load endpoints
        if endpoints:
            self.endpoints = endpoints
        elif pool_path:
            self._load_from_file(pool_path)
        else:
            raise ValueError("Either pool_path or endpoints must be provided")
        
        if not self.endpoints:
            raise ValueError("No endpoints loaded")
        
        # initialize status for each endpoint
        for endpoint in self.endpoints:
            if endpoint not in self.endpoint_status:
                self.endpoint_status[endpoint] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'total_response_time': 0.0,
                    'last_error': None,
                }
        
        print(f"RerankerEndpointPool initialized with {len(self.endpoints)} endpoints")
        for i, endpoint in enumerate(self.endpoints):
            print(f"  [{i+1}] {endpoint}")
    
    def _load_from_file(self, pool_path: str):
        """load endpoints from file"""
        path = Path(pool_path)
        if not path.is_absolute():
            # try to find file relative to shared/configs/
            project_root = Path(__file__).parent.parent.parent
            path = project_root / "shared" / "configs" / pool_path
        
        if not path.exists():
            raise FileNotFoundError(f"Reranker endpoint pool file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.endpoints = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # ensure URL format is correct
                if not line.startswith('http://') and not line.startswith('https://'):
                    line = f"http://{line}"
                self.endpoints.append(line)
    
    def get_endpoint(self) -> str:
        """
        Get next available endpoint (round-robin strategy)
        
        Returns:
            Available endpoint URL
        """
        with self.lock:
            if not self.endpoints:
                raise ValueError("No reranker endpoints available in pool")
            
            if self.use_round_robin:
                # Round-robin: simple round-robin, ensure uniform distribution
                selected_idx = self.current_index
                self.current_index = (self.current_index + 1) % len(self.endpoints)
                
                selected_endpoint = self.endpoints[selected_idx]
                self.endpoint_status[selected_endpoint]['total_requests'] += 1
                
                return selected_endpoint
            else:
                # smart selection mode (can select based on error rate, etc.)
                # simple implementation: select endpoint with least requests
                min_requests = min(
                    self.endpoint_status[ep]['total_requests']
                    for ep in self.endpoints
                )
                candidates = [
                    ep for ep in self.endpoints
                    if self.endpoint_status[ep]['total_requests'] == min_requests
                ]
                selected_endpoint = candidates[0]
                self.endpoint_status[selected_endpoint]['total_requests'] += 1
                return selected_endpoint
    
    def mark_success(self, endpoint: str, response_time: float = 0.0):
        """mark endpoint request as successful"""
        with self.lock:
            if endpoint in self.endpoint_status:
                self.endpoint_status[endpoint]['successful_requests'] += 1
                self.endpoint_status[endpoint]['total_response_time'] += response_time
    
    def mark_error(self, endpoint: str, error: str):
        """mark endpoint request as failed"""
        with self.lock:
            if endpoint in self.endpoint_status:
                self.endpoint_status[endpoint]['failed_requests'] += 1
                self.endpoint_status[endpoint]['last_error'] = error
    
    def get_status(self) -> Dict:
        """get pool status information"""
        with self.lock:
            endpoints_status = {}
            for endpoint, status in self.endpoint_status.items():
                total = status['total_requests']
                success = status['successful_requests']
                failed = status['failed_requests']
                avg_time = (
                    status['total_response_time'] / success
                    if success > 0 else 0.0
                )
                
                endpoints_status[endpoint] = {
                    'total_requests': total,
                    'successful_requests': success,
                    'failed_requests': failed,
                    'success_rate': success / total if total > 0 else 0.0,
                    'avg_response_time': avg_time,
                    'last_error': status['last_error'],
                }
            
            return {
                'total_endpoints': len(self.endpoints),
                'endpoints_status': endpoints_status,
            }
