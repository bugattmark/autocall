# aws_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse
import os
import time
from subprocess import run
from funcs import f_asr, f_llm, f_tts
from collections import defaultdict, deque

app = FastAPI()

_short_term_mem = defaultdict(lambda: deque(maxlen=6))

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load your prompt template from disk, once at startup:
with open(os.path.join(os.path.dirname(__file__), "prompt.txt"), encoding="utf-8") as f:
    RAW_PROMPT = f.read()

# 2) Fill in the business‐specific values (change these or load dynamically):
business_name    = "บอทน้อย"
appointment_date = "10 กรกฎาคม"
appointment_time = "15:30"


FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

@app.post("/process")
async def process_audio(file: UploadFile = File(...), call_id: str = Header("default")):
    # 1) read & save incoming bytes
    data = await file.read()
    if not data:
        raise HTTPException(400, "No audio data received")
    print(f">>> [/process] got {len(data)} bytes", flush=True)

    # 2) write to disk
    ext    = file.filename.rsplit(".", 1)[-1].lower()
    raw_fn = f"received_raw.{ext}"
    with open(raw_fn, "wb") as f:
        f.write(data)

    # 3) convert to 16k mono WAV if needed
    wav_fn = raw_fn
    if ext != "wav":
        wav_fn = "received.wav"
        run(
            [FFMPEG_PATH, "-y", "-i", raw_fn, "-ar", "16000", "-ac", "1", wav_fn],
            check=True
        )
        os.remove(raw_fn)

    # 4) ASR
    t0 = time.time()
    text = await f_asr(wav_fn)
    dur_asr = time.time() - t0

    # ── build one big prompt string from SYSTEM_PROMPT + memory + current turn ──
    mem = _short_term_mem[call_id]

    # serialize only past turns
    history_lines = []
    for turn in mem:
        who = "User" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{who}: {turn['content']}")

    # now substitute into the raw template
    full_prompt = RAW_PROMPT.format(
        BUSINESS_NAME = business_name,
        APPOINTMENT_DATE = appointment_date,
        APPOINTMENT_TIME = appointment_time,
        RECENT_DIALOG = "\n".join(history_lines),
        CURRENT_UTTERANCE = text
    )
 
    # only now store this turn for next time
    mem.append({"role": "user", "content": text})

    # debug‐print the exact prompt sent to the LLM
    print("=== LLM PROMPT START ===\n" + full_prompt + "\n=== LLM PROMPT END ===", flush=True)

    # call LLM
    t1 = time.time()
    reply = await f_llm(full_prompt, "")
    dur_llm = time.time() - t1

    # store the assistant reply
    mem.append({"role": "assistant", "content": reply})
    # ─────────────────────────────────────────────────────────────────────────

    # 6) TTS → writes "response.wav"
    out_wav = await f_tts(reply, filename="response.wav")

    # 7) logging
    print(f"[ASR ] {text} ({dur_asr:.2f}s)", flush=True)
    print(f"[LLM ] {reply} ({dur_llm:.2f}s)", flush=True)

    # 8) cleanup
    os.remove(wav_fn)

    # 9) return the TTS audio
    return FileResponse(out_wav, media_type="audio/wav")
