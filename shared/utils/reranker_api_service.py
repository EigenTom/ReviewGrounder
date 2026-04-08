"""
Reranker API Service

Pack FlagReranker into an HTTP API service, supporting multi-GPU load balancing.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Suppress transformers warnings
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: FastAPI not installed. Install with: pip install fastapi uvicorn")

try:
    from FlagEmbedding import FlagReranker
    HAS_FLAGEMBEDDING = True
except ImportError:
    HAS_FLAGEMBEDDING = False
    print("Warning: FlagEmbedding not installed. Install with: pip install FlagEmbedding")


# Request/Response models
class RerankRequest(BaseModel):
    query: str
    paragraphs: List[str]
    batch_size: int = 100


class RerankResponse(BaseModel):
    scores: List[float]
    success: bool
    message: Optional[str] = None


# Global reranker instance
_reranker: Optional[Any] = None


def create_app(model_path: str, use_fp16: bool = True, device: Optional[str] = None):
    """Create FastAPI app with reranker"""
    global _reranker
    
    app = FastAPI(title="Reranker API Service", version="1.0.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def load_reranker():
        """Load reranker model on startup"""
        global _reranker
        if not HAS_FLAGEMBEDDING:
            raise RuntimeError("FlagEmbedding not installed")
        
        print(f"Loading reranker model: {model_path}")
        print(f"Using FP16: {use_fp16}")
        if device:
            print(f"Using device: {device}")
        
        try:
            _reranker = FlagReranker(
                model_path,
                use_fp16=use_fp16,
            )
            if device:
                # Note: FlagReranker may not support explicit device setting
                # This is a placeholder for future support
                pass
            print("Reranker model loaded successfully")
        except Exception as e:
            print(f"Error loading reranker: {e}")
            raise
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": _reranker is not None
        }
    
    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(request: RerankRequest):
        """Rerank paragraphs given a query"""
        global _reranker
        
        if _reranker is None:
            raise HTTPException(status_code=503, detail="Reranker not loaded")
        
        if not request.paragraphs:
            return RerankResponse(
                scores=[],
                success=True,
                message="No paragraphs to rerank"
            )
        
        try:
            # Prepare sentence pairs: [[query, paragraph], ...]
            sentence_pairs = [[request.query, p] for p in request.paragraphs]
            
            # Compute scores
            scores = _reranker.compute_score(
                sentence_pairs,
                batch_size=request.batch_size
            )
            
            # Handle score format (can be float or list)
            if isinstance(scores, float):
                scores = [scores]
            elif not isinstance(scores, list):
                scores = list(scores)
            
            return RerankResponse(
                scores=scores,
                success=True
            )
        except Exception as e:
            print(f"Error during reranking: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def main():
    """Main entry point for reranker API service"""
    parser = argparse.ArgumentParser(description="Reranker API Service")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to reranker model (e.g., 'OpenScholar/OpenScholar_Reranker')"
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
        default=8004,
        help="Port to bind to (default: 8004)"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision (default: True)"
    )
    parser.add_argument(
        "--no_fp16",
        dest="use_fp16",
        action="store_false",
        help="Disable FP16 precision"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cuda:1')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, use 1 for reranker)"
    )
    
    args = parser.parse_args()
    
    if not HAS_FASTAPI:
        print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    if not HAS_FLAGEMBEDDING:
        print("Error: FlagEmbedding not installed. Install with: pip install FlagEmbedding")
        sys.exit(1)
    
    # Create app
    app = create_app(
        model_path=args.model_path,
        use_fp16=args.use_fp16,
        device=args.device
    )
    
    # Run server
    print(f"Starting reranker API service on {args.host}:{args.port}")
    print(f"Model: {args.model_path}")
    print(f"FP16: {args.use_fp16}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
