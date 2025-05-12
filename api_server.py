
import os
import torch
import soundfile as sf
import logging
import argparse
import platform
import uvicorn

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI


# Configure logging
logging.basicConfig(level=logging.INFO)


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device_id=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")

    # # Determine appropriate device based on platform and availability
    if platform.system() == "Darwin":
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{device_id}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{device_id}")
        logging.info(f"Using CUDA device: {device}")
    else:
        #     # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    try:
        model = SparkTTS(model_dir, device)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load SparkTTS model from {model_dir}")


def run_tts(
    text: str,
    model: SparkTTS,
    prompt_text: Optional[str] = None,
    prompt_speech_path: Optional[str] = None,
    gender: Optional[str] = None,
    pitch: Optional[str] = None,
    speed: Optional[str] = None,
    save_dir: str = "example/results_api",
) -> str:
    """Perform TTS inference and save the generated audio. Returns the path to the audio file."""
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")
    save_path = os.path.join(save_dir, f"tts_output_{timestamp}.wav")

    logging.info("Starting TTS inference...")

    try:
        with torch.no_grad():
            wav = model.inference(
                text,
                prompt_speech_path,
                prompt_text,
                gender,
                pitch,
                speed,
            )
            sf.write(save_path, wav, samplerate=16000)
        logging.info(f"Audio saved at: {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Error during TTS inference: {e}")
        # Consider removing the partially created file if an error occurs
        if os.path.exists(save_path):
            os.remove(save_path)
        raise RuntimeError(f"TTS inference failed: {e}")


def parse_arguments():
    """
    Parse command-line arguments such as model directory and device ID.
    """
    parser = argparse.ArgumentParser(description="Spark TTS API server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU/MPS device to use (e.g., 0 for cuda:0 or mps:0)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the API server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="example/results_api",
        help="Directory to save generated audio files."
    )
    return parser.parse_args()


app = FastAPI()

# Global variable to hold the model and output directory
model_tts: Optional[SparkTTS] = None
output_audio_dir: str = "example/results_api"

class TTSCreateRequest(BaseModel):
    text: str
    gender: str  # 'male' or 'female'
    pitch: int   # 1-5
    speed: int   # 1-5

@app.on_event("startup")
async def startup_event():
    """Load the model and set up configurations when the server starts."""
    global model_tts, output_audio_dir
    args = parse_arguments()
    model_tts = initialize_model(model_dir=args.model_dir, device_id=args.device)
    output_audio_dir = args.output_dir
    # Ensure output directory exists
    os.makedirs(output_audio_dir, exist_ok=True)
    logging.info(f"API server started. Model loaded. Output directory: {output_audio_dir}")

@app.post("/tts/create", response_class=FileResponse)
async def api_create_voice(
    text: str = Form(...),
    gender: str = Form(...),
    pitch: int = Form(...),
    speed: int = Form(...)
):
    """
    Create a synthetic voice with adjustable parameters.
    Returns the generated WAV audio file.
    """
    if not model_tts:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again shortly.")
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Invalid gender. Choose 'male' or 'female'.")
    if not (1 <= pitch <= 5):
        raise HTTPException(status_code=400, detail="Invalid pitch. Choose an integer between 1 and 5.")
    if not (1 <= speed <= 5):
        raise HTTPException(status_code=400, detail="Invalid speed. Choose an integer between 1 and 5.")

    try:
        pitch_val = LEVELS_MAP_UI[pitch]
        speed_val = LEVELS_MAP_UI[speed]

        audio_output_path = run_tts(
            text=text,
            model=model_tts,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val,
            save_dir=output_audio_dir
        )
        return FileResponse(path=audio_output_path, media_type='audio/wav', filename=os.path.basename(audio_output_path))
    except RuntimeError as e:
        logging.error(f"TTS creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError:
        # This can happen if pitch/speed int is not in LEVELS_MAP_UI, though validation should catch it.
        logging.error(f"Invalid pitch or speed value not found in LEVELS_MAP_UI.")
        raise HTTPException(status_code=400, detail="Invalid pitch or speed value.")

@app.post("/tts/clone", response_class=FileResponse)
async def api_clone_voice(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    prompt_text: Optional[str] = Form(None)
):
    """
    Clone voice using text and a prompt audio file.
    Returns the generated WAV audio file.
    """
    if not model_tts:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again shortly.")

    # Save the uploaded prompt audio to a temporary file
    temp_prompt_path = os.path.join(output_audio_dir, f"prompt_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}_{prompt_audio.filename}")
    try:
        with open(temp_prompt_path, "wb") as buffer:
            buffer.write(await prompt_audio.read())
        logging.info(f"Prompt audio saved to {temp_prompt_path}")

        audio_output_path = run_tts(
            text=text,
            model=model_tts,
            prompt_text=prompt_text,
            prompt_speech_path=temp_prompt_path,
            save_dir=output_audio_dir
        )
        return FileResponse(path=audio_output_path, media_type='audio/wav', filename=os.path.basename(audio_output_path))
    except RuntimeError as e:
        logging.error(f"TTS cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary prompt file
        if os.path.exists(temp_prompt_path):
            try:
                os.remove(temp_prompt_path)
                logging.info(f"Cleaned up temporary prompt audio: {temp_prompt_path}")
            except OSError as e:
                logging.error(f"Error deleting temporary prompt audio {temp_prompt_path}: {e}")


if __name__ == "__main__":
    args = parse_arguments()

    # Note: The model is loaded via the startup_event when uvicorn starts the app.
    # This avoids loading the model twice if __name__ == "__main__" is run directly by some tools.

    uvicorn.run(app, host=args.host, port=args.port) 