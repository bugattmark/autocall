from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse
import os
import time
from subprocess import run
from funcs import f_asr, f_llm, f_tts
from collections import defaultdict, deque

app = FastAPI()

_short_term_mem = defaultdict(lambda: deque(maxlen=6))

with open(os.path.join(os.path.dirname(__file__), "prompt.txt"), encoding="utf-8") as f:
    RAW_PROMPT = f.read()

business_name    = "A"
appointment_date = "10 กรกฎาคม"
appointment_time = "15:30"


FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

@app.post("/process")
async def process_audio(file: UploadFile = File(...), call_id: str = Header("default")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "No audio data received")
    print(f">>> [/process] got {len(data)} bytes", flush=True)

    ext    = file.filename.rsplit(".", 1)[-1].lower()
    raw_fn = f"received_raw.{ext}"
    with open(raw_fn, "wb") as f:
        f.write(data)

    wav_fn = raw_fn
    if ext != "wav":
        wav_fn = "received.wav"
        run(
            [FFMPEG_PATH, "-y", "-i", raw_fn, "-ar", "16000", "-ac", "1", wav_fn],
            check=True
        )
        os.remove(raw_fn)

    t0 = time.time()
    text = await f_asr(wav_fn)
    dur_asr = time.time() - t0

    mem = _short_term_mem[call_id]

    history_lines = []
    for turn in mem:
        who = "User" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{who}: {turn['content']}")

    full_prompt = RAW_PROMPT.format(
        BUSINESS_NAME = business_name,
        APPOINTMENT_DATE = appointment_date,
        APPOINTMENT_TIME = appointment_time,
        RECENT_DIALOG = "\n".join(history_lines),
        CURRENT_UTTERANCE = text
    )
 
    mem.append({"role": "user", "content": text})

    print("=== LLM PROMPT START ===\n" + full_prompt + "\n=== LLM PROMPT END ===", flush=True)

    t1 = time.time()
    reply = await f_llm(full_prompt, "")
    dur_llm = time.time() - t1

    mem.append({"role": "assistant", "content": reply})

    out_wav = await f_tts(reply, filename="response.wav")

    print(f"[ASR ] {text} ({dur_asr:.2f}s)", flush=True)
    print(f"[LLM ] {reply} ({dur_llm:.2f}s)", flush=True)

    try:
        os.remove(wav_fn)
    except OSError:
        pass

    return FileResponse(out_wav, media_type="audio/wav")
