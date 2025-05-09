# whisper-danger-zone
How to do speech to text

```
pip install -r requirements.txt
```

and the usage

```bash
python main.py audio.mp3 --diarize --hf-token $HUGGINGFACE_TOKEN -o transcript.txt
```

or

```bash
python main.py audio.wav > out.txt
```