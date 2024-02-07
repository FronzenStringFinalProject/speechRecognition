import re
import tempfile

import wave
from io import BytesIO

import cn2an
import numpy
import uvicorn
import whisper
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated

model = whisper.load_model("medium", device="cuda", download_root="./model", in_memory=True)

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_credentials=True,
                   allow_methods=["*"])
pattern = re.compile(r"(\d+)")


@app.post("/recognition")
def recognize(file: Annotated[bytes, File()]):
    with wave.open(BytesIO(file), "rb") as wav:
        sample_rate = wav.getframerate()
        audio = wav.readframes(sample_rate)
        audio_as_np_u16 = numpy.frombuffer(audio, dtype=numpy.int16)
        float_audio = audio_as_np_u16.astype(numpy.float32)

        audio_normalized = float_audio/2**15
    result = model.transcribe(
        audio_normalized,
        language="zh",
        initial_prompt="以下是用普通话的句子回答数学题目答案，有可能不会做。"
    )
    print(result)

    text = result["text"]
    text = cn2an.transform(text)
    ans_str = pattern.findall(text)
    if ans_str:
        ans_str = int(ans_str[-1])
    else:
        ans_str = None
    unknow = False
    for w in ["不会", "不知道","好难"]:
        if w in text:
            unknow = True
    return {"result": ans_str, "unknown": unknow}

if __name__ == '__main__':
    uvicorn.run(app,port=5000)