import zipfile
import io
import json
from utils.recognition import Recognition
from fastapi import FastAPI, Request

app = FastAPI()

unzip_folder = "utils/unzip_files"


def unzip(files):
    unzip_files = zipfile.ZipFile(io.BytesIO(files), mode='r')
    unzip_files.extractall(unzip_folder)

@app.post("/")
async def root(request: Request):
    files: bytes = await request.body()
    unzip(files)
    df = Recognition.recognize(unzip_folder)
    df = df.reset_index(drop=True)
    df_json = df.to_json(orient='columns')
    response = json.loads(df_json)
    return response