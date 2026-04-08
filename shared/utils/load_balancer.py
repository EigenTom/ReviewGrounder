"""
Simple Python Load Balancer

A lightweight load balancer for vLLM and Reranker services.
Uses FastAPI to forward requests to multiple backend services.
"""
import os
import sys
import time
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Optional, Any
from threading import Lock
import argparse

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import StreamingResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: FastAPI not installed. Install with: pip install fastapi uvicorn httpx")


class SimpleLoadBalancer:
    """Simple load balancer with round-robin and least-connection strategies"""
    
    def __init__(
        self,
        backends: List[str],
        strategy: str = "round_robin",
        health_check_interval: float = 10.0,
    ):
        """
        Initialize load balancer
        
        Args:
            backends: List of backend URLs (e.g., ["http://localhost:8000", "http://localhost:8001"])
            strategy: Load balancing strategy ("round_robin" or "least_conn")
            health_check_interval: Health check interval in seconds
        """
        self.backends = backends
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        # Round-robin state
        self.current_index = 0
        self.index_lock = Lock()
        
        # Least-connection state
        self.connection_counts: Dict[str, int] = {backend: 0 for backend in backends}
        self.conn_lock = Lock()
        
        # Health check state
        self.healthy_backends: Dict[str, bool] = {backend: True for backend in backends}
        self.health_lock = Lock()
        
        # HTTP client for forwarding requests
        self.client = httpx.AsyncClient(timeout=300.0)
        
        print(f"Load balancer initialized with {len(backends)} backends")
        print(f"Strategy: {strategy}")
        for i, backend in enumerate(backends):
            print(f"  [{i+1}] {backend}")
    
    def get_backend(self) -> Optional[str]:
        """Get next backend based on strategy"""
        with self.health_lock:
            available_backends = [b for b in self.backends if self.healthy_backends.get(b, True)]
        
        if not available_backends:
            # If no healthy backends, try all backends
            available_backends = self.backends
        
        if not available_backends:
            return None
        
        if self.strategy == "round_robin":
            with self.index_lock:
                backend = available_backends[self.current_index % len(available_backends)]
                self.current_index = (self.current_index + 1) % len(available_backends)
            return backend
        
        elif self.strategy == "least_conn":
            with self.conn_lock:
                # Find backend with least connections
                backend = min(available_backends, key=lambda b: self.connection_counts.get(b, 0))
                self.connection_counts[backend] = self.connection_counts.get(backend, 0) + 1
            return backend
        
        else:
            # Default to round-robin
            with self.index_lock:
                backend = available_backends[self.current_index % len(available_backends)]
                self.current_index = (self.current_index + 1) % len(available_backends)
            return backend
    
    def release_backend(self, backend: str):
        """Release a backend (for least-conn strategy)"""
        if self.strategy == "least_conn":
            with self.conn_lock:
                self.connection_counts[backend] = max(0, self.connection_counts.get(backend, 0) - 1)
    
    async def health_check(self, backend: str) -> bool:
        """Check if a backend is healthy"""
        try:
            # For vLLM backends (URLs ending with /v1), use /models endpoint
            # For other backends, try /health first, then root
            if backend.endswith("/v1"):
                # vLLM endpoint: try /models (which becomes /v1/models)
                endpoints = ["/models", "/"]
            else:
                # Other services: try /health, then root
                endpoints = ["/health", "/"]
            
            for endpoint in endpoints:
                try:
                    response = await self.client.get(f"{backend}{endpoint}", timeout=5.0)
                    if response.status_code < 500:
                        return True
                except:
                    continue
            return False
        except Exception:
            return False
    
    async def check_all_backends(self):
        """Check health of all backends"""
        while True:
            for backend in self.backends:
                is_healthy = await self.health_check(backend)
                with self.health_lock:
                    was_healthy = self.healthy_backends.get(backend, True)
                    self.healthy_backends[backend] = is_healthy
                if not is_healthy:
                    print(f"Warning: Backend {backend} is unhealthy")
                elif not was_healthy:
                    print(f"Info: Backend {backend} recovered (healthy again)")
            await asyncio.sleep(self.health_check_interval)
    
    async def forward_request(
        self,
        method: str,
        path: str,
        request: Request,
        backend: Optional[str] = None
    ) -> Response:
        """Forward a request to a backend"""
        if backend is None:
            backend = self.get_backend()
        
        if backend is None:
            raise HTTPException(status_code=503, detail="No healthy backends available")
        
        try:
            # Get request body
            body = await request.body()
            
            # Get query parameters
            query_params = dict(request.query_params)
            
            # Get headers (exclude host and connection)
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("connection", None)
            headers.pop("content-length", None)
            
            # Forward request
            url = f"{backend}{path}"
            if query_params:
                url += "?" + "&".join(f"{k}={v}" for k, v in query_params.items())
            
            response = await self.client.request(
                method=method,
                url=url,
                content=body,
                headers=headers,
            )
            
            # Create response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        
        except Exception as e:
            # Mark backend as unhealthy
            with self.health_lock:
                self.healthy_backends[backend] = False
            
            self.release_backend(backend)
            raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")
        finally:
            self.release_backend(backend)
    
    async def forward_streaming_request(
        self,
        method: str,
        path: str,
        request: Request,
        backend: Optional[str] = None
    ):
        """Forward a streaming request to a backend"""
        if backend is None:
            backend = self.get_backend()
        
        if backend is None:
            raise HTTPException(status_code=503, detail="No healthy backends available")
        
        try:
            # Get request body
            body = await request.body()
            
            # Get query parameters
            query_params = dict(request.query_params)
            
            # Get headers
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("connection", None)
            headers.pop("content-length", None)
            
            # Forward request
            url = f"{backend}{path}"
            if query_params:
                url += "?" + "&".join(f"{k}={v}" for k, v in query_params.items())
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    method=method,
                    url=url,
                    content=body,
                    headers=headers,
                ) as response:
                    async def generate():
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    
                    return StreamingResponse(
                        generate(),
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )
        
        except Exception as e:
            # Mark backend as unhealthy
            with self.health_lock:
                self.healthy_backends[backend] = False
            
            self.release_backend(backend)
            raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")
        finally:
            self.release_backend(backend)


def create_load_balancer_app(
    backends: List[str],
    strategy: str = "round_robin",
    health_check_interval: float = 10.0,
) -> FastAPI:
    """Create FastAPI app with load balancer"""
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed. Install with: pip install fastapi uvicorn httpx")
    
    app = FastAPI(title="Simple Load Balancer", version="1.0.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create load balancer
    lb = SimpleLoadBalancer(backends, strategy, health_check_interval)
    
    # Start health check task
    @app.on_event("startup")
    async def start_health_check():
        asyncio.create_task(lb.check_all_backends())
    
    # Health check endpoint
    @app.get("/health")
    async def health():
        healthy_count = sum(1 for h in lb.healthy_backends.values() if h)
        return {
            "status": "healthy" if healthy_count > 0 else "unhealthy",
            "healthy_backends": healthy_count,
            "total_backends": len(lb.backends),
            "backends": [
                {
                    "url": backend,
                    "healthy": lb.healthy_backends.get(backend, False),
                    "connections": lb.connection_counts.get(backend, 0) if lb.strategy == "least_conn" else None,
                }
                for backend in lb.backends
            ]
        }
    
    # Forward all other requests
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def forward(request: Request, path: str):
        method = request.method
        is_streaming = "stream" in request.query_params or "stream=true" in str(request.url)
        
        if is_streaming:
            return await lb.forward_streaming_request(method, f"/{path}", request)
        else:
            return await lb.forward_request(method, f"/{path}", request)
    
    return app


def main():
    """Main entry point for load balancer"""
    parser = argparse.ArgumentParser(description="Simple Python Load Balancer")
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        required=True,
        help="Backend URLs (space-separated or single comma-separated string)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="round_robin",
        choices=["round_robin", "least_conn"],
        help="Load balancing strategy (default: round_robin)"
    )
    parser.add_argument(
        "--health-check-interval",
        type=float,
        default=10.0,
        help="Health check interval in seconds (default: 10.0)"
    )
    
    args = parser.parse_args()
    # Allow comma-separated single string (reliable when invoked from shell nohup)
    if len(args.backends) == 1 and "," in args.backends[0]:
        args.backends = [u.strip() for u in args.backends[0].split(",") if u.strip()]
    
    if not HAS_FASTAPI:
        print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn httpx")
        sys.exit(1)
    
    # Create app
    app = create_load_balancer_app(
        backends=args.backends,
        strategy=args.strategy,
        health_check_interval=args.health_check_interval,
    )
    
    # Run server
    print(f"Starting load balancer on {args.host}:{args.port}")
    print(f"Strategy: {args.strategy}")
    print(f"Backends: {', '.join(args.backends)}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

# python -m shared.utils.load_balancer --backends http://localhost:8000 http://localhost:8001 http://localhost:8002 http://localhost:8003     --port 8004     --strategy round_robin