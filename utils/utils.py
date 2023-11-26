import glob
import os
import uuid
from typing import List

from pydub import AudioSegment

from .helpers import (
    convert_numbers_to_words,
    download_large_file_from_hugging_face,
    generate_audio,
    initialize_inference,
    preprocess_english,
    remove_empty_lines,
    save_audio,
)

# global values for infrence
device = None
style_encoder = None
generator = None
tokenizer = None
token2id = None
speaker2id = None


def generate_unique_filename(extension=".wav"):
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename


def generate_phoneme_from_text(text: str) -> List[str]:
    """
    Main function to generate phonemes from text.
    Args:
        text (str): input text

    Returns:
        list(str): list of phonemes
    """

    phonemes = []

    for line in text.splitlines():
        phonemes.append(preprocess_english(line.rstrip()))

    return phonemes


def process_text(input_text: str) -> str:
    """
    Run simple preprocessing on a text.
    From my testing replacing numbers with words improves the quality of the output.
    Leaving empty lines caused phonems generation to fail.

    Args:
        input_text (str): input text
    Returns:
        str: processed text
    """
    return convert_numbers_to_words(remove_empty_lines(input_text))


def add_speaker_and_emotion(text: List[str], speaker: str = "9017", emotion: str = "Happy") -> str:
    """
    Add speaker and emotion to the prompt.
    Args:
        text (str): input text
        speaker (str): speaker ID
        emotion (str): emotion

    Returns:
        str: prompt with speaker and emotion
    """

    proccessed_text = []

    for line in text:
        proccessed_text.append(
            speaker + "|" + emotion + "|" + line + "|Emoti-Voice - a Multi-Voice and Prompt-Controlled T-T-S Engine"
        )

    return "\n".join(proccessed_text)


def download_model_checkpoints(
    models_path: str = f"{os.getcwd()}/outputs/prompt_tts_open_source_joint/ckpt",
    models: list = ["do_00140000", "g_00140000", "checkpoint_163431"],
    hugging_face_url: str = "https://huggingface.co/FPVG/EmotiVoice-pretrained-models/resolve/main/",
) -> None:
    """
    Download the model checkpoints.
    Args:
        model_path (str): path to save the checkpoints
    """
    for model in models:
        if model == "checkpoint_163431":
            save_path = f"{os.getcwd()}/outputs/style_encoder/ckpt"
        else:
            save_path = f"{models_path}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_url = hugging_face_url + model + "?download=true"

        download_large_file_from_hugging_face(model, model_url, save_path)


def load_models(logdir: str = f"prompt_tts_open_source_joint", checkpoint: str = f"g_00140000") -> None:
    global device, style_encoder, generator, tokenizer, token2id, speaker2id

    device, style_encoder, generator, tokenizer, token2id, speaker2id = initialize_inference(logdir, checkpoint)


def run_tts(
    processed_text: str,
    output_path: str = f"{os.getcwd()}/outputs/prompt_tts_open_source_joint/test_audio/audio",
    output_filename: str = "my_audio.wav",
) -> None:
    global device, style_encoder, generator, tokenizer, token2id, speaker2id

    if not device:
        load_models()

    audio = generate_audio(processed_text, device, style_encoder, generator, tokenizer, token2id, speaker2id)
    save_audio(output_path, output_filename, audio, sample_rate=16000)
