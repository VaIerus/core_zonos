#!/usr/bin/env python3
"""
Startup script for Zonos TTS API Server.

This script handles the initialization and startup of the Zonos FastAPI server
with proper environment setup and error handling.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're in the correct virtual environment."""
    try:
        # Check if we're in the Zonos virtual environment
        venv_path = Path(".env_win")
        if not venv_path.exists():
            logger.error("Zonos virtual environment not found (.env_win)")
            return False
        
        # Check if required packages are available
        try:
            import torch
            import torchaudio
            import fastapi
            import uvicorn
            logger.info("All required packages found")
            return True
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        return False

def check_models():
    """Check if Zonos models are available."""
    models_dir = Path("models")

    if not models_dir.exists():
        logger.warning("Models directory not found")
        return False

    # Check for transformer model in multiple locations
    model_found = False

    # Direct paths
    transformer_model = models_dir / "Zyphra--Zonos-v0.1-transformer"
    hybrid_model = models_dir / "Zyphra--Zonos-v0.1-hybrid"

    if transformer_model.exists() and (transformer_model / "config.json").exists():
        logger.info("Transformer model found (direct path)")
        model_found = True
    elif hybrid_model.exists() and (hybrid_model / "config.json").exists():
        logger.info("Hybrid model found (direct path)")
        model_found = True

    # Check HuggingFace cache paths
    if not model_found:
        hf_cache_dir = models_dir / "hf_download" / "hub"
        if hf_cache_dir.exists():
            # Check transformer model in HF cache
            transformer_hf = hf_cache_dir / "models--Zyphra--Zonos-v0.1-transformer"
            if transformer_hf.exists():
                snapshots_dir = transformer_hf / "snapshots"
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        if snapshot_dir.is_dir() and (snapshot_dir / "config.json").exists():
                            logger.info(f"Transformer model found in HF cache: {snapshot_dir}")
                            model_found = True
                            break

            # Check hybrid model in HF cache
            if not model_found:
                hybrid_hf = hf_cache_dir / "models--Zyphra--Zonos-v0.1-hybrid"
                if hybrid_hf.exists():
                    snapshots_dir = hybrid_hf / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot_dir in snapshots_dir.iterdir():
                            if snapshot_dir.is_dir() and (snapshot_dir / "config.json").exists():
                                logger.info(f"Hybrid model found in HF cache: {snapshot_dir}")
                                model_found = True
                                break

    if not model_found:
        logger.warning("No Zonos models found in models directory or HF cache")
        logger.info("Expected locations:")
        logger.info("  - models/Zyphra--Zonos-v0.1-transformer/")
        logger.info("  - models/hf_download/hub/models--Zyphra--Zonos-v0.1-transformer/snapshots/*/")

    return model_found

def check_speaker_audio():
    """Check if speaker audio file exists."""
    speaker_path = Path("../voices/Nepshort.mp3")
    
    if speaker_path.exists():
        logger.info(f"Speaker audio found: {speaker_path}")
        return True
    else:
        logger.warning(f"Speaker audio not found: {speaker_path}")
        return False

def start_server(host="127.0.0.1", port=8000, reload=False):
    """Start the Zonos API server."""
    try:
        logger.info("Starting Zonos TTS API Server...")
        logger.info(f"Server will be available at: http://{host}:{port}")
        logger.info("API documentation will be available at: http://{host}:{port}/docs")
        
        # Import and run the server
        import uvicorn
        
        uvicorn.run(
            "zonos_api_server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

def main():
    """Main startup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Zonos TTS API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment and model checks")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Zonos TTS API Server Startup")
    logger.info("=" * 60)
    
    if not args.skip_checks:
        logger.info("Performing startup checks...")
        
        # Check environment
        if not check_environment():
            logger.error("Environment check failed. Please ensure you're in the Zonos virtual environment.")
            sys.exit(1)
        
        # Check models
        if not check_models():
            logger.warning("Model check failed. Server may not function properly.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Check speaker audio
        check_speaker_audio()
        
        logger.info("Startup checks completed")
    
    # Start the server
    start_server(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
