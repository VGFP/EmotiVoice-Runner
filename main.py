import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.utils import (
    add_speaker_and_emotion,
    download_model_checkpoints,
    generate_phoneme_from_text,
    generate_unique_filename,
    load_models,
    process_text,
    run_tts,
)


class TTSRequest(BaseModel):
    text: str
    speaker: str = "9017"
    emotion: str = "Happy"


# Download the model checkpoints
download_model_checkpoints()

# Load the models
load_models()

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/generate_audio/")
async def generate_audio(request: TTSRequest):
    try:
        # Process the request data
        processed_text = process_text(request.text)
        phoneme = generate_phoneme_from_text(processed_text)
        processed_phoneme = add_speaker_and_emotion(phoneme, request.speaker, request.emotion)

        # Generate the audio file
        output_path = f"{os.getcwd()}/outputs/prompt_tts_open_source_joint/test_audio/audio"
        output_filename = generate_unique_filename()
        run_tts(processed_phoneme, output_path, output_filename)

        # Return the audio file
        file_path = f"{output_path}/{output_filename}"

        torch.cuda.empty_cache()

        return FileResponse(path=file_path, media_type="audio/wav", filename=output_filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
