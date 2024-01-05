# EmotiVoice-Runner
Code for easier setting up interface for EmotiVoice Text to Speach model

## Installation

```bash
pip install -r requirements.txt
pip install torch torchaudio
```

To run FastAPI server: 

```bash
uvicorn main:app --reload
```

Loading make take a while becaues script will download models from Hugging face if they are not already downloaded.

# Based on EmotiVoice 
https://github.com/netease-youdao/EmotiVoice
https://github.com/netease-youdao/EmotiVoice/blob/main/LICENSE
