import argparse
import os
import re
import sys
from string import punctuation

import numpy as np
from g2p_en import G2p

sys.path.insert(0, os.getcwd())

import argparse
import glob
import os
import sys
import uuid
import warnings
from typing import Dict, List, Tuple

import numpy as np
import requests
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from yacs import config as CONFIG

from config.joint.config import Config
from models.hifigan.get_vocoder import MAX_WAV_VALUE
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder


def generate_unique_filename(extension=".wav"):
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text):
    lexicon = read_lexicon(f"{os.getcwd()}/utils/lexicon/librispeech-lexicon.txt")

    g2p = G2p()
    phones = []
    words = list(filter(lambda x: x not in {"", " "}, re.split(r"([,;.\-\?\!\s+])", text)))

    for w in words:
        if w.lower() in lexicon:
            phones += ["[" + ph + "]" for ph in lexicon[w.lower()]] + ["engsp1"]
        else:
            phone = g2p(w)
            if not phone:
                continue

            if phone[0].isalnum():
                phones += ["[" + ph + "]" if ph != " " else "engsp1" for ph in phone]
            elif phone == " ":
                continue
            else:
                phones.pop()  # pop engsp1
                phones.append("engsp4")
    if "engsp" in phones[-1]:
        phones.pop()

    mark = "." if text[-1] != "?" else "?"
    phones = ["<sos/eos>"] + phones + [mark, "<sos/eos>"]
    return " ".join(phones)


def remove_empty_lines(text):
    # Split the text into lines
    lines = text.split("\n")

    # Filter out empty or whitespace-only lines
    non_empty_lines = [line for line in lines if line.strip()]

    # Join the remaining lines back into a single string
    return "\n".join(non_empty_lines)


def convert_numbers_to_words(text):
    import re

    from num2words import num2words

    # Function to convert a single number to words
    def number_to_word(match):
        return num2words(match.group())

    # Regular expression to identify numbers in the text
    number_pattern = r"\b\d+\b"

    # Replace all occurrences of numbers with their word form
    return re.sub(number_pattern, number_to_word, text)


def download_large_file_from_hugging_face(
    file_name: str, hugging_face_url: str, local_directory: str, chunk_size: int = 1024 * 1024
) -> None:
    """
    Downloads a large file from Hugging Face if it is not already present in the local directory.
    The download is done in chunks to handle large files efficiently.

    :param file_name: Name of the file to download.
    :param hugging_face_url: URL of the file on Hugging Face.
    :param local_directory: Directory to check for the file and save it if not present.
    :param chunk_size: Size of each chunk in bytes. Default is 1MB.
    :return: None
    """
    local_file_path = os.path.join(local_directory, file_name)

    # Check if the file already exists
    if not os.path.exists(local_file_path):
        try:
            # Stream the download
            with requests.get(hugging_face_url, stream=True) as r:
                r.raise_for_status()
                with open(local_file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        # filter out keep-alive chunks
                        if chunk:
                            f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_name}. Error: {e}")


def initialize_inference(
    logdir: str, checkpoint: str
) -> Tuple[torch.device, nn.Module, nn.Module, PreTrainedTokenizer, Dict[str, int], Dict[str, int]]:
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config.model_config_path, "r") as fin:
        conf = CONFIG.load_cfg(fin)

    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config).to(device)
    model_CKPT = torch.load(
        os.path.join(os.getcwd(), "outputs", "style_encoder", "ckpt", "checkpoint_163431"), map_location="cpu"
    )
    model_ckpt = {}
    for key, value in model_CKPT["model"].items():
        new_key = key[7:]
        model_ckpt[new_key] = value
    style_encoder.load_state_dict(model_ckpt)

    generator = JETSGenerator(conf).to(device)
    ckpt_path = os.path.join(os.path.join(os.getcwd(), "outputs"), logdir, "ckpt", checkpoint)
    model_CKPT = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(model_CKPT["generator"])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    with open(config.token_list_path, "r") as f:
        token2id = {t.strip(): idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding="utf-8") as f:
        speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

    return device, style_encoder, generator, tokenizer, token2id, speaker2id


def generate_audio(
    text: str,
    device: torch.device,
    style_encoder: nn.Module,
    generator: nn.Module,
    tokenizer: PreTrainedTokenizer,
    token2id: Dict[str, int],
    speaker2id: Dict[str, int],
) -> np.ndarray:
    audio_arr = []

    texts = []
    prompts = []
    speakers = []
    contents = []

    for line in text.split("\n"):
        line = line.strip().split("|")
        speakers.append(line[0])
        prompts.append(line[1])
        texts.append(line[2].split())
        contents.append(line[3])

    for i, (speaker, prompt, text, content) in enumerate(tqdm(zip(speakers, prompts, texts, contents))):
        style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
        content_embedding = get_style_embedding(content, tokenizer, style_encoder)

        if speaker not in speaker2id:
            continue
        speaker = speaker2id[speaker]
        text_int = [token2id[ph] for ph in text]

        sequence = torch.from_numpy(np.array(text_int)).to(device).long().unsqueeze(0)
        sequence_len = torch.from_numpy(np.array([len(text_int)])).to(device)
        style_embedding = torch.from_numpy(style_embedding).to(device).unsqueeze(0)
        content_embedding = torch.from_numpy(content_embedding).to(device).unsqueeze(0)
        speaker = torch.from_numpy(np.array([speaker])).to(device)
        with torch.no_grad():
            infer_output = generator(
                inputs_ling=sequence,
                inputs_style_embedding=style_embedding,
                input_lengths=sequence_len,
                inputs_content_embedding=content_embedding,
                inputs_speaker=speaker,
                alpha=1.0,
            )
            audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
            audio_arr.append(audio)

    return np.concatenate(audio_arr)


def save_audio(output_path: str, output_filename: str, audio_data: np.ndarray, sample_rate: int) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    sf.write(os.path.join(output_path, output_filename), data=audio_data, samplerate=sample_rate)


def get_style_embedding(prompt, tokenizer, style_encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"].to(device)
    token_type_ids = prompt["token_type_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)

    with torch.no_grad():
        output = style_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding
