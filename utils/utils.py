from helpers import preprocess_english, remove_empty_lines, convert_numbers_to_words, generate_audio, initialize_inference, save_audio

from pydub import AudioSegment
import glob
import os

# global values for infrence
device = None
style_encoder = None
generator = None
tokenizer = None
token2id = None
speaker2id = None

def generate_phoneme_from_text(text: str) -> list(str):
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

"""
def combine_wav_files(
    folder_path: str = f"{os.getcwd()}/outputs/prompt_tts_open_source_joint/test_audio/audio/g_00140000",
    output_file: str = f"{os.getcwd()}/combined_output1.wav",
    delete_originals: bool = False
) -> None:
    """
    Combine all WAV files in a folder into a single WAV file and optionally delete the original files.

    :param folder_path: Path to the folder containing WAV files.
    :param output_file: Path of the output combined WAV file.
    :param delete_originals: If True, delete the original WAV files after combining them.
    """

    # Find all WAV files and sort them by their numerical value
    wav_files = sorted(
        glob.glob(os.path.join(folder_path, "*.wav")), key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )

    # Combine WAV files
    combined = AudioSegment.empty()
    for wav_file in wav_files:
        sound = AudioSegment.from_wav(wav_file)
        combined += sound

    # Export combined audio
    combined.export(output_file, format="wav")

    # Optionally delete the original WAV files
    if delete_originals:
        for wav_file in wav_files:
            os.remove(wav_file)
"""

def add_speaker_and_emotion(speaker: str = "9017", emotion: str = "Happy") -> str:
    """
    Add speaker and emotion to the prompt.
    Args:
        speaker (str): speaker ID
        emotion (str): emotion

    Returns:
        str: prompt with speaker and emotion
    """

    proccessed_text = []

    for line in emotions.splitlines():
        proccessed_text.append(
            speaker + "|" + emotion + "|" + line + "|Emoti-Voice - a Multi-Voice and Prompt-Controlled T-T-S Engine"
        )
    
    return "\n".join(proccessed_text)


def download_model_checkpoints(models_path: str = f"{os.getcwd()}/outputs/prompt_tts_open_source_joint/ckpt", models: list = ["do_00140000", "g_00140000", "checkpoint_163431"], hugging_face_url: str = "https://huggingface.co/FPVG/EmotiVoice-pretrained-models/blob/main/") -> None:
    """
    Download the model checkpoints.
    Args:
        model_path (str): path to save the checkpoints
    """
    
    for model in models:
        model_path = os.path.join(models_path, model)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_url = hugging_face_url + model

        download_large_file_from_hugging_face(model, hugging_face_url, model_path)


def load_models( logdir: str = f"prompt_tts_open_source_joint", checkpoint: str = f"g_00140000") -> None:
    global device, style_encoder, generator, tokenizer, token2id, speaker2id

    device, style_encoder, generator, tokenizer, token2id, speaker2id = initialize_inference(logdir, checkpoint)


def run_tts(processed_text: str,  output_path: str = f"{os.getcwd()}/outputs/prompt_tts_open_source_joint/test_audio/audio/my_audio.wav") -> None:
    global device, style_encoder, generator, tokenizer, token2id, speaker2id

    if not device:
        load_models()

    audio = generate_audio(processed_text, device, style_encoder, generator, tokenizer, token2id, speaker2id)
    save_audio(audio, output_path)