import whisper
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated

model = whisper.load_model("medium", device="cuda", download_root="./model", in_memory=True)

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_credentials=True,
                   allow_methods=["*"])


@app.post("/recognition")
def recognize(file: Annotated[bytes, File()]):
    with open("tmp/record.wav", "wb") as tmp:
        tmp.write(file)
        tmp.flush()

    result = model.transcribe("tmp/record.wav", language="zh", initial_prompt="以下是普通话的句子。")
    print(result)

    return {"result": result["text"]}


if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
