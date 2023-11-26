import os
import sys

sys.path.insert(0, os.getcwd())

from utils.utils import (
    add_speaker_and_emotion,
    download_model_checkpoints,
    generate_phoneme_from_text,
    load_models,
    process_text,
    run_tts,
)

"""
Proper workflow:
download model checkpoints -> load_models -> process_text -> generate_phoneme_from_text -> add_speaker_and_emotion -> run_tts
"""


def run_example():
    text = """Sure, I can add type hints to the code for better clarity and to leverage Python's type checking capabilities. Type hints help in understanding what types of arguments a function expects and what type it returns, making the code more maintainable and less prone to errors.
    
    By dividing the function into smaller parts, you gain the flexibility of loading models and configurations only once and reusing them for each inference call. This approach is more efficient and easier to maintain, especially for repeated inference tasks."""

    # download model checkpoints
    download_model_checkpoints()

    # load models
    load_models()

    # process text
    processed_text = process_text(input_text=text)

    # generate phoneme from text
    phoneme = generate_phoneme_from_text(processed_text)

    # add speaker and emotion
    processed_phoneme = add_speaker_and_emotion(phoneme)

    # run tts
    run_tts(processed_phoneme)

    print("Done!")


if __name__ == "__main__":
    run_example()
