#!/usr/bin/env python3
"""
FastAPI Server for Zonos TTS Integration with RavenV2.

This server provides a REST API interface to Zonos TTS functionality,
allowing RavenV2 to generate speech with PAD-driven emotional prosody
while keeping Zonos isolated in its own virtual environment.

Key Features:
- PAD emotion vector to Zonos 8-dimensional emotion mapping
- Voice cloning with configurable reference audio
- Prosody parameter mapping (pitch, rate, etc.)
- Audio generation and file management
- Health checks and status monitoring
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import base64

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add Zonos to path
sys.path.insert(0, str(Path(__file__).parent))

# Zonos imports
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
CURRENT_MODEL = None
SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None

# API Models
class PADState(BaseModel):
    """PAD emotional state from RavenV2."""
    pleasure: float = Field(..., ge=-1.0, le=1.0, description="Pleasure value (-1 to 1)")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="Arousal value (-1 to 1)")
    dominance: float = Field(..., ge=-1.0, le=1.0, description="Dominance value (-1 to 1)")
    source: str = Field(default="api", description="Source of PAD values")

class TTSRequest(BaseModel):
    """TTS generation request from RavenV2."""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
    pad_state: Optional[PADState] = Field(None, description="PAD emotional state")
    tone_name: Optional[str] = Field(None, description="Tone name from MotivationManager")
    voice_profile: str = Field(default="base_voice", description="Voice profile name")
    language: str = Field(default="en-us", description="Language code")
    priority: int = Field(default=0, description="Request priority")
    
    # Zonos-specific parameters (optional overrides)
    pitch_std: Optional[float] = Field(None, ge=0.0, le=100.0, description="Pitch variation")
    speaking_rate: Optional[float] = Field(None, ge=0.1, le=3.0, description="Speaking rate")
    cfg_scale: Optional[float] = Field(None, ge=1.0, le=10.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class TTSResponse(BaseModel):
    """TTS generation response."""
    success: bool
    audio_file: Optional[str] = None
    audio_base64: Optional[str] = None
    generation_time: float
    request_id: str
    pad_state: Optional[PADState] = None
    prosody_params: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    model_loaded: bool
    speaker_loaded: bool
    speaker_audio_path: Optional[str]
    device: str
    supported_languages: List[str]
    uptime: float

# FastAPI app
app = FastAPI(
    title="Zonos TTS API Server",
    description="FastAPI bridge for Zonos TTS integration with RavenV2",
    version="1.0.0"
)

# Server startup time
SERVER_START_TIME = time.time()

def pad_to_zonos_emotions(pleasure: float, arousal: float, dominance: float) -> Dict[str, float]:
    """
    Convert PAD values to Zonos emotion parameters (e1-e8) with proper mapping.

    Based on Zonos emotion vector format and typical emotional mappings:
    - e1: Happiness/Joy (high pleasure, moderate arousal)
    - e2: Sadness (low pleasure, low arousal)
    - e3: Disgust (low pleasure, moderate dominance)
    - e4: Fear (low pleasure, high arousal, low dominance)
    - e5: Surprise (neutral pleasure, high arousal)
    - e6: Anger (low pleasure, high arousal, high dominance)
    - e7: Other/Mixed emotions
    - e8: Neutral baseline

    Args:
        pleasure: Pleasure value (-1 to 1)
        arousal: Arousal value (-1 to 1)
        dominance: Dominance value (-1 to 1)

    Returns:
        Dictionary with e1-e8 emotion values that sum to 1.0
    """
    # Initialize emotion values
    emotions = {
        "e1": 0.05,  # happiness - baseline
        "e2": 0.05,  # sadness - baseline
        "e3": 0.05,  # disgust - baseline
        "e4": 0.05,  # fear - baseline
        "e5": 0.05,  # surprise - baseline
        "e6": 0.05,  # anger - baseline
        "e7": 0.1,   # other - baseline
        "e8": 0.6    # neutral - dominant baseline
    }

    # Map pleasure to happiness/sadness axis
    if pleasure > 0.2:
        # High pleasure -> happiness
        emotions["e1"] = 0.3 + (pleasure * 0.5)  # 0.3 to 0.8
        emotions["e8"] = max(0.1, emotions["e8"] - pleasure * 0.3)
    elif pleasure < -0.2:
        # Low pleasure -> sadness
        emotions["e2"] = 0.2 + (abs(pleasure) * 0.4)  # 0.2 to 0.6
        emotions["e8"] = max(0.1, emotions["e8"] - abs(pleasure) * 0.3)

    # Map arousal and dominance combinations
    if arousal > 0.5 and dominance > 0.4:
        # High arousal + high dominance -> anger
        emotions["e6"] = 0.3 + (arousal * dominance * 0.4)
        emotions["e8"] = max(0.1, emotions["e8"] - 0.2)
    elif arousal > 0.6 and dominance < -0.3:
        # High arousal + low dominance -> fear
        emotions["e4"] = 0.2 + (arousal * abs(dominance) * 0.3)
        emotions["e8"] = max(0.1, emotions["e8"] - 0.15)
    elif arousal > 0.5 and abs(dominance) < 0.3:
        # High arousal + neutral dominance -> surprise
        emotions["e5"] = 0.2 + (arousal * 0.3)
        emotions["e8"] = max(0.1, emotions["e8"] - 0.1)
    elif pleasure < -0.3 and dominance > 0.2:
        # Low pleasure + moderate dominance -> disgust
        emotions["e3"] = 0.15 + (abs(pleasure) * dominance * 0.2)
        emotions["e8"] = max(0.1, emotions["e8"] - 0.1)

    # Handle mixed/complex emotions
    emotion_intensity = (abs(pleasure) + abs(arousal) + abs(dominance)) / 3
    if emotion_intensity > 0.6:
        emotions["e7"] = min(0.3, 0.1 + emotion_intensity * 0.2)
        emotions["e8"] = max(0.1, emotions["e8"] - 0.1)

    # Normalize to sum to 1.0
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}
    else:
        # Fallback to neutral
        emotions = {"e1": 0.05, "e2": 0.05, "e3": 0.05, "e4": 0.05,
                   "e5": 0.05, "e6": 0.05, "e7": 0.1, "e8": 0.6}

    return emotions

def map_pad_to_zonos_params(pad_state: PADState, tone_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Map PAD state to complete Zonos generation parameters.

    Args:
        pad_state: PAD emotional state
        tone_name: Optional tone name for adjustments

    Returns:
        Dictionary of all Zonos parameters for audio generation
    """
    # Get emotion mapping
    emotions = pad_to_zonos_emotions(pad_state.pleasure, pad_state.arousal, pad_state.dominance)

    # Base prosody parameters
    base_pitch_std = 45.0  # Match Zonos default
    base_speaking_rate = 15.0  # Match Zonos default

    # Map PAD to prosody adjustments
    pleasure = pad_state.pleasure
    arousal = pad_state.arousal
    dominance = pad_state.dominance

    # Calculate prosody adjustments based on PAD
    pitch_adjustment = dominance * 0.4 + arousal * 0.3  # Higher dominance/arousal = higher pitch variation
    rate_adjustment = arousal * 0.5 + pleasure * 0.3    # Higher arousal/pleasure = faster speech

    # Apply tone-specific adjustments
    if tone_name:
        if tone_name in ["flirty", "playful"]:
            pitch_adjustment += 0.3
            rate_adjustment += 0.2
            emotions["e1"] = min(1.0, emotions["e1"] + 0.1)  # Boost happiness
        elif tone_name in ["supportive", "caring"]:
            pitch_adjustment -= 0.2
            rate_adjustment -= 0.1
            emotions["e8"] = min(1.0, emotions["e8"] + 0.1)  # Boost neutral/calm
        elif tone_name in ["confident", "assertive"]:
            pitch_adjustment += 0.2
            rate_adjustment += 0.3
            emotions["e6"] = min(1.0, emotions["e6"] + 0.05)  # Slight anger/assertiveness

    # Calculate final prosody parameters
    pitch_std = max(20.0, min(80.0, base_pitch_std + pitch_adjustment * 15))
    speaking_rate = max(8.0, min(25.0, base_speaking_rate + rate_adjustment * 6))

    # Build complete Zonos parameter set
    zonos_params = {
        # Emotion parameters (e1-e8)
        **emotions,

        # Prosody parameters
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,

        # Audio quality parameters
        "vq_single": 0.78,  # Voice quality
        "fmax": 24000,      # Max frequency
        "dnsmos_ovrl": 4,   # DNS-MOS overall quality
        "speaker_noised": False,

        # Generation parameters
        "cfg_scale": 4.0,   # Classifier-free guidance scale
        "top_p": 0.8,       # Nucleus sampling
        "top_k": 200,       # Top-k sampling
        "min_p": 0.15,      # Minimum probability
        "linear": 0.0,      # Linear interpolation
        "confidence": 0.0,  # Confidence threshold
        "quadratic": 0.0,   # Quadratic interpolation

        # Randomization
        "seed": 420,        # Default seed
        "randomize_seed": False,

        # Unconditional generation control
        # CRITICAL: unconditional_keys must NOT contain "emotion" to enable emotion conditioning
        # If "emotion" is in unconditional_keys, the model will ignore e1-e8 emotion parameters
        "unconditional_keys": [],  # Empty array ensures emotion conditioning is enabled
        "disable_torch_compile": True  # Keep as per user requirements
    }

    # Validate that emotion conditioning is enabled
    if "emotion" in zonos_params["unconditional_keys"]:
        logger.warning("CRITICAL: 'emotion' found in unconditional_keys - emotion conditioning will be DISABLED!")
        logger.warning("Removing 'emotion' from unconditional_keys to enable emotion conditioning")
        zonos_params["unconditional_keys"] = [key for key in zonos_params["unconditional_keys"] if key != "emotion"]

    logger.info(f"Emotion conditioning ENABLED - unconditional_keys: {zonos_params['unconditional_keys']}")

    return zonos_params

async def load_model_and_speaker(model_choice: str = "transformer", speaker_audio_path: str = None):
    """Load Zonos model and speaker embedding."""
    global CURRENT_MODEL, SPEAKER_EMBEDDING, SPEAKER_AUDIO_PATH

    try:
        logger.info(f"Loading Zonos model: {model_choice}")

        # Get the script directory to ensure correct paths
        script_dir = Path(__file__).parent
        logger.info(f"Script directory: {script_dir}")

        # Try multiple possible model locations
        possible_paths = []

        if model_choice == "transformer":
            # Direct path (if models were extracted)
            possible_paths.append(script_dir / "models/Zyphra--Zonos-v0.1-transformer")
            # HuggingFace cache path
            hf_cache_base = script_dir / "models/hf_download/hub/models--Zyphra--Zonos-v0.1-transformer"
            if hf_cache_base.exists():
                # Find the snapshot directory
                snapshots_dir = hf_cache_base / "snapshots"
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        if snapshot_dir.is_dir():
                            possible_paths.append(snapshot_dir)
                            break
        else:
            # Hybrid model paths
            possible_paths.append(script_dir / "models/Zyphra--Zonos-v0.1-hybrid")
            hf_cache_base = script_dir / "models/hf_download/hub/models--Zyphra--Zonos-v0.1-hybrid"
            if hf_cache_base.exists():
                snapshots_dir = hf_cache_base / "snapshots"
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        if snapshot_dir.is_dir():
                            possible_paths.append(snapshot_dir)
                            break

        # Try each possible path
        model_dir = None
        for path in possible_paths:
            config_path = path / "config.json"
            model_path = path / "model.safetensors"

            logger.info(f"Checking path: {path}")
            logger.info(f"  Config exists: {config_path.exists()}")
            logger.info(f"  Model exists: {model_path.exists()}")

            if config_path.exists() and model_path.exists():
                model_dir = path
                logger.info(f"Found model files in: {model_dir}")
                break

        if model_dir is None:
            raise FileNotFoundError(f"Model files not found in any of these locations: {possible_paths}")

        config_path = model_dir / "config.json"
        model_path = model_dir / "model.safetensors"

        # Load model
        CURRENT_MODEL = Zonos.from_local(str(config_path), str(model_path), device=device)
        logger.info(f"Model loaded successfully on {device}")

        # Load speaker embedding
        if speaker_audio_path:
            # Handle relative paths from script directory
            if not Path(speaker_audio_path).is_absolute():
                speaker_path = script_dir / speaker_audio_path
            else:
                speaker_path = Path(speaker_audio_path)

            if speaker_path.exists():
                logger.info(f"Loading speaker audio: {speaker_path}")
                wav, sampling_rate = torchaudio.load(str(speaker_path))
                SPEAKER_EMBEDDING = CURRENT_MODEL.make_speaker_embedding(wav, sampling_rate)
                SPEAKER_AUDIO_PATH = str(speaker_path)
                logger.info("Speaker embedding created successfully")
            else:
                logger.warning(f"Speaker audio not found: {speaker_path}")
        else:
            logger.warning("No speaker audio path provided")

    except Exception as e:
        logger.error(f"Failed to load model/speaker: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    logger.info("Starting Zonos TTS API Server...")
    
    # Default speaker audio path (can be overridden via API)
    default_speaker_path = "../voices/Nepshort.mp3"
    
    try:
        await load_model_and_speaker("transformer", default_speaker_path)
        logger.info("Server startup completed successfully")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        # Continue startup even if model loading fails - can be retried via API

@app.get("/", response_model=ServerStatus)
async def root():
    """Get server status and health check."""
    uptime = time.time() - SERVER_START_TIME
    
    return ServerStatus(
        status="running",
        model_loaded=CURRENT_MODEL is not None,
        speaker_loaded=SPEAKER_EMBEDDING is not None,
        speaker_audio_path=SPEAKER_AUDIO_PATH,
        device=str(device),
        supported_languages=supported_language_codes,
        uptime=uptime
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/tts/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from text with PAD-driven emotional prosody.

    This is the main endpoint that RavenV2 will call to generate speech.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Validate model is loaded
        if CURRENT_MODEL is None:
            raise HTTPException(status_code=503, detail="TTS model not loaded")

        if SPEAKER_EMBEDDING is None:
            raise HTTPException(status_code=503, detail="Speaker embedding not loaded")

        logger.info(f"TTS request {request_id}: '{request.text[:50]}...'")

        # Map PAD to complete Zonos parameters
        if request.pad_state:
            zonos_params = map_pad_to_zonos_params(request.pad_state, request.tone_name)
            logger.info(f"PAD mapping: P={request.pad_state.pleasure:.2f}, A={request.pad_state.arousal:.2f}, D={request.pad_state.dominance:.2f}")
            logger.info(f"Emotion vector: e1={zonos_params['e1']:.3f}, e2={zonos_params['e2']:.3f}, e3={zonos_params['e3']:.3f}, e4={zonos_params['e4']:.3f}")
            logger.info(f"                e5={zonos_params['e5']:.3f}, e6={zonos_params['e6']:.3f}, e7={zonos_params['e7']:.3f}, e8={zonos_params['e8']:.3f}")
        else:
            # Use neutral defaults
            neutral_pad = PADState(pleasure=0.0, arousal=0.0, dominance=0.0)
            zonos_params = map_pad_to_zonos_params(neutral_pad, request.tone_name)

        # Override with request-specific parameters if provided
        if request.pitch_std is not None:
            zonos_params["pitch_std"] = request.pitch_std
        if request.speaking_rate is not None:
            zonos_params["speaking_rate"] = request.speaking_rate
        if request.cfg_scale is not None:
            zonos_params["cfg_scale"] = request.cfg_scale
        if request.seed is not None:
            zonos_params["seed"] = request.seed
            torch.manual_seed(request.seed)

        # Use the already-loaded model directly instead of calling generate_audio
        logger.info(f"Generating audio with Zonos parameters:")
        logger.info(f"  Prosody: pitch_std={zonos_params['pitch_std']:.1f}, speaking_rate={zonos_params['speaking_rate']:.1f}")
        logger.info(f"  Generation: cfg_scale={zonos_params['cfg_scale']:.1f}, seed={zonos_params['seed']}")

        # Create emotion tensor from PAD-mapped e1-e8 values
        emotion_vector = [
            zonos_params["e1"], zonos_params["e2"], zonos_params["e3"], zonos_params["e4"],
            zonos_params["e5"], zonos_params["e6"], zonos_params["e7"], zonos_params["e8"]
        ]
        emotion_tensor = torch.tensor(emotion_vector, dtype=torch.float32, device=device)

        # Create VQ tensor (voice quality)
        vq_tensor = torch.tensor([zonos_params["vq_single"]] * 8, device=device).unsqueeze(0)

        # Create conditioning dictionary with ALL parameters including emotion
        cond_dict = make_cond_dict(
            text=request.text,
            speaker=SPEAKER_EMBEDDING,
            language=request.language,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=zonos_params["fmax"],
            pitch_std=zonos_params["pitch_std"],
            speaking_rate=zonos_params["speaking_rate"],
            dnsmos_ovrl=zonos_params["dnsmos_ovrl"],
            speaker_noised=zonos_params["speaker_noised"],
            device=device,
            unconditional_keys=zonos_params["unconditional_keys"]  # CRITICAL: Empty list enables emotion conditioning
        )

        logger.info(f"Emotion conditioning: emotion_tensor={emotion_tensor.tolist()}")
        logger.info(f"Unconditional keys: {zonos_params['unconditional_keys']} (empty = emotion enabled)")

        # Prepare conditioning
        conditioning = CURRENT_MODEL.prepare_conditioning(cond_dict)

        # Prepare sampling parameters for Zonos
        sampling_params = {
            "min_p": zonos_params["min_p"],
            "top_p": zonos_params["top_p"],
            "top_k": zonos_params["top_k"]
        }

        # Generate audio codes with Zonos parameters
        codes = CURRENT_MODEL.generate(
            conditioning,
            cfg_scale=zonos_params["cfg_scale"],
            sampling_params=sampling_params,
            disable_torch_compile=zonos_params["disable_torch_compile"]
        )

        # Decode codes to audio using the autoencoder
        audio = CURRENT_MODEL.autoencoder.decode(codes).cpu().detach()

        # Handle multi-channel audio (take first channel if multiple)
        if audio.dim() == 2 and audio.size(0) > 1:
            audio = audio[0:1, :]

        # Store the final seed used
        final_seed = zonos_params["seed"]

        # Save audio file
        output_dir = Path("../outputs/tts")
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_filename = f"tts_{request_id}_{int(time.time())}.wav"
        audio_path = output_dir / audio_filename

        # Save audio using the correct sample rate from the autoencoder
        if isinstance(audio, torch.Tensor):
            # Get the sample rate from the autoencoder
            sample_rate = CURRENT_MODEL.autoencoder.sampling_rate

            # Handle tensor dimensions properly
            logger.info(f"Audio tensor shape before processing: {audio.shape}")

            # Squeeze to remove batch dimension if present, then ensure 2D for torchaudio.save
            if audio.dim() == 3:  # [batch, channels, samples]
                audio = audio.squeeze(0)  # Remove batch dimension -> [channels, samples]
            elif audio.dim() == 1:  # [samples]
                audio = audio.unsqueeze(0)  # Add channel dimension -> [1, samples]
            # If already 2D [channels, samples], keep as is

            logger.info(f"Audio tensor shape after processing: {audio.shape}")

            # Save the audio file
            torchaudio.save(str(audio_path), audio, sample_rate)
            logger.info(f"Audio saved: {audio_path} ({sample_rate} Hz, {audio.shape})")
        else:
            raise ValueError(f"Unexpected audio format: {type(audio)}")

        generation_time = time.time() - start_time

        logger.info(f"TTS generation completed in {generation_time:.2f}s: {audio_path}")

        # Schedule cleanup of old files
        background_tasks.add_task(cleanup_old_files, output_dir)

        return TTSResponse(
            success=True,
            audio_file=str(audio_path),
            generation_time=generation_time,
            request_id=request_id,
            pad_state=request.pad_state,
            prosody_params={
                "emotions": {k: v for k, v in zonos_params.items() if k.startswith('e')},
                "prosody": {
                    "pitch_std": zonos_params["pitch_std"],
                    "speaking_rate": zonos_params["speaking_rate"],
                    "cfg_scale": zonos_params["cfg_scale"]
                },
                "generation": {
                    "seed": final_seed if 'final_seed' in locals() else zonos_params["seed"],
                    "top_p": zonos_params["top_p"],
                    "top_k": zonos_params["top_k"],
                    "min_p": zonos_params["min_p"]
                }
            }
        )

    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = f"TTS generation failed: {str(e)}"
        logger.error(f"Request {request_id}: {error_msg}")

        return TTSResponse(
            success=False,
            generation_time=generation_time,
            request_id=request_id,
            pad_state=request.pad_state,
            error=error_msg
        )

@app.get("/tts/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files."""
    audio_path = Path("../outputs/tts") / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=filename
    )

@app.post("/speaker/load")
async def load_speaker(speaker_audio_path: str):
    """Load a new speaker embedding from audio file."""
    global SPEAKER_EMBEDDING, SPEAKER_AUDIO_PATH

    try:
        if CURRENT_MODEL is None:
            raise HTTPException(status_code=503, detail="TTS model not loaded")

        audio_path = Path(speaker_audio_path)
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"Speaker audio file not found: {speaker_audio_path}")

        logger.info(f"Loading new speaker: {speaker_audio_path}")
        wav, sampling_rate = torchaudio.load(str(audio_path))
        SPEAKER_EMBEDDING = CURRENT_MODEL.make_speaker_embedding(wav, sampling_rate)
        SPEAKER_AUDIO_PATH = speaker_audio_path

        return {"success": True, "speaker_path": speaker_audio_path, "message": "Speaker loaded successfully"}

    except Exception as e:
        error_msg = f"Failed to load speaker: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/model/load")
async def load_model(model_choice: str = "transformer"):
    """Load or reload the TTS model."""
    try:
        await load_model_and_speaker(model_choice, SPEAKER_AUDIO_PATH)
        return {"success": True, "model": model_choice, "message": "Model loaded successfully"}

    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def cleanup_old_files(output_dir: Path, max_age_hours: int = 24):
    """Clean up old audio files to prevent disk space issues."""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for file_path in output_dir.glob("tts_*.wav"):
            if current_time - file_path.stat().st_mtime > max_age_seconds:
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path}")

    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zonos TTS API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    logger.info(f"Starting Zonos TTS API Server on {args.host}:{args.port}")

    uvicorn.run(
        "zonos_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
