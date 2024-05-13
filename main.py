import base64
import re
import wave
from io import BytesIO
from typing import Optional

import cn2an
import numpy
import pydantic
import uvicorn
import whisper
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated

model = whisper.load_model("large", device="cuda", download_root="./model", in_memory=True)

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_credentials=True,
                   allow_methods=["*"])
pattern = re.compile(r"(\d+)")


class InputItem(pydantic.BaseModel):
    base64_voice: str
    ans_num: int = 10


class Result(pydantic.BaseModel):
    result: Optional[int]
    negative: bool
    positive: bool


@app.post("/recognition")
def recognize(item: InputItem):
    file = base64.b64decode(item.base64_voice.encode("utf-8"))
    with wave.open(BytesIO(file), "rb") as wav:
        sample_rate = wav.getframerate()
        audio = wav.readframes(sample_rate)
        audio_as_np_u16 = numpy.frombuffer(audio, dtype=numpy.int16)
        float_audio = audio_as_np_u16.astype(numpy.float32)

        audio_normalized = float_audio / 2 ** 15
    result = model.transcribe(
        audio_normalized,
        language="Chinese",
        initial_prompt="以下是用普通话的句子回答数学题目答案。"
    )
    print(result)
    ret = Result(result=None, negative=False, positive=False)

    text = result["text"]
    text = cn2an.transform(text)
    text = cn2an.transform(text)
    print(text)
    ans_str = pattern.findall(text)

    if ans_str:
        ret.result = int(ans_str[-1])

    for w in ["不会", "不知道", "好难", "不", "不需要"]:
        if w in text:
            ret.negative = True
    for w in ["对", "没错", "需要", "嗯"]:
        if w in text:
            ret.positive = True
    return ret


if __name__ == '__main__':
    uvicorn.run(app, port=5000)
