from utils.utils import generate_phoneme_from_text, add_speaker_and_emotion, process_text, download_model_checkpoints, run_tts, load_models

"""
Proper workflow:
download model checkpoints -> load_models -> process_text -> generate_phoneme_from_text -> add_speaker_and_emotion -> run_tts
"""



if __name__ == '__main__':
    main()

def main():

    text = """Sure, I can add type hints to the code for better clarity and to leverage Python's type checking capabilities. Type hints help in understanding what types of arguments a function expects and what type it returns, making the code more maintainable and less prone to errors.
    
    By dividing the function into smaller parts, you gain the flexibility of loading models and configurations only once and reusing them for each inference call. This approach is more efficient and easier to maintain, especially for repeated inference tasks."""

    # download model checkpoints
    download_model_checkpoints()

    # load models
    load_models()

    # process text
    processed_text = process_text(text = text)

    # generate phoneme from text
    phoneme = generate_phoneme_from_text(processed_text)

    # add speaker and emotion
    processed_phoneme = add_speaker_and_emotion(phoneme)

    # run tts
    run_tts(processed_phoneme)

    print("Done!")
