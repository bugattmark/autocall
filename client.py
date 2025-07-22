# client.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import HTMLResponse
import httpx

app = FastAPI()

REMOTE_API = "..."  #ngrok api url

@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
<html>
  <body>
    <h3>Hold to Talk & Hear the Reply</h3>
    <button id="talkBtn" style="font-size:1.2em;padding:1em">
      🎤
    </button>
    <audio id="player" controls style="width:100%;margin-top:1em"></audio>

    <script>
    let mediaRecorder, chunks = [], audioBlob;
    const talkBtn = document.getElementById('talkBtn');
    const player  = document.getElementById('player');

    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = e => chunks.push(e.data);
        mediaRecorder.onstop = () => {
          audioBlob = new Blob(chunks, { type: 'audio/webm' });
          sendAudio(audioBlob, 'recording.webm');
        };
      })
      .catch(e => alert('Mic error: ' + e.message));

    talkBtn.addEventListener('mousedown', () => {
      chunks = [];
      mediaRecorder.start();
      talkBtn.textContent = 'Listening…';
    });

    talkBtn.addEventListener('mouseup', () => {
      mediaRecorder.stop();
      talkBtn.textContent = '🎤';
    });

    async function sendAudio(blob, name) {
      const fd = new FormData();
      fd.append('file', blob, name);
      let resp;
      try {
        resp = await fetch('/upload', { method:'POST', body: fd });
        resp.raiseForStatus?.();
      } catch (e) {
        return alert('Network error: ' + e.message);
      }
      if (!resp.ok) {
        const txt = await resp.text();
        return alert(`Error ${resp.status}: ${txt}`);
      }
      const buf  = await resp.arrayBuffer();
      const type = resp.headers.get('content-type') || 'audio/wav';
      const out  = new Blob([buf], { type });
      player.src = URL.createObjectURL(out);
      player.load();
      player.play().catch(console.warn);
    }
    </script>
  </body>
</html>
"""

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    #forward verbatim to aws_server `/process`
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(REMOTE_API, files={"file": (file.filename, file.file, file.content_type)})
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Upstream error: {e}")

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
    )