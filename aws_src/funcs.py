import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pythaitts import TTS

#### ASR
MODEL_NAME = "biodatlab/whisper-th-medium-combined"  #see alternative model names below
lang = "th"
device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=15,
    device=device,
)

#### LLM
model_id = "scb10x/typhoon2.1-gemma3-4b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

### TTS
tts = TTS()

import asyncio

async def f_asr(file: str) -> str:
    def _sync_asr():
        return pipe(
            file,
            generate_kwargs={
                "language": "<|th|>",
                "task": "transcribe",
                "max_length": 225,
                "no_repeat_ngram_size": 3,
                "temperature": 0.0,
                "eos_token_id": pipe.model.config.eos_token_id
            },
            batch_size=16
        )["text"]
    return await asyncio.to_thread(_sync_asr)

async def f_tts(text: str, filename: str = "answer.wav") -> str:
    def _sync_tts():
        try:
            return tts.tts(text, filename=filename)
        except KeyError as e:
            bad = e.args[0]
            cleaned = text.replace(bad, "")
            return tts.tts(cleaned, filename=filename)
    return await asyncio.to_thread(_sync_tts)

async def f_llm(prompt: str, query: str) -> str:
    def _sync_llm():
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": query},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            early_stopping=True,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        resp_tokens = outputs[0][input_ids.shape[-1]:]
        return clean_thai_text(tokenizer.decode(resp_tokens, skip_special_tokens=True))
    return await asyncio.to_thread(_sync_llm)

import re

_digit = {
    0: "ศูนย์", 1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่",
    5: "ห้า",   6: "หก",   7: "เจ็ด", 8: "แปด", 9: "เก้า"
}

def thai_number(n: int) -> str:
    """Convert 0 <= n <= 1_000_000 into Thai words."""
    if n < 0 or n > 1_000_000:
        raise ValueError("Supported range is 0 to 1,000,000")
    if n == 0:
        return _digit[0]
    if n == 1_000_000:
        return "หนึ่งล้าน"

    def _sub_th(x: int) -> str:
        s = ""
        # แสน (100_000)
        if x >= 100_000:
            h = x // 100_000
            if h > 1:
                s += _sub_th(h)
            s += "แสน"
            x %= 100_000
        # หมื่น (10_000)
        if x >= 10_000:
            t = x // 10_000
            if t > 1:
                s += _sub_th(t)
            s += "หมื่น"
            x %= 10_000
        # พัน (1_000)
        if x >= 1_000:
            th = x // 1_000
            if th > 1:
                s += _sub_th(th)
            s += "พัน"
            x %= 1_000
        # ร้อย (100)
        if x >= 100:
            h = x // 100
            if h > 1:
                s += _sub_th(h)
            s += "ร้อย"
            x %= 100
        # สิบ (10)
        if x >= 10:
            t = x // 10
            if t > 1:
                s += _digit[t]
            s += "สิบ"
            x %= 10
        # หน่วย
        if x > 0:
            if x == 1 and s:
                s += "เอ็ด"
            else:
                s += _digit[x]
        return s

    return _sub_th(n)

def clean_thai_text(text: str) -> str:
    def _replace_num(match: re.Match) -> str:
        num = int(match.group())
        if 0 <= num <= 1_000_000:
            return thai_number(num)
        return "".join(_digit[int(d)] for d in match.group())
    text = re.sub(r"\d+", _replace_num, text)
    text = text.replace("\n", " ")
    return "".join(ch for ch in text if "\u0E00" <= ch <= "\u0E7F" or ch.isspace())

