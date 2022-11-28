import zipfile
import io
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
    response = df.reset_index().to_json(orient='records')[1:-1].replace('},{', '} {')
    return {response}