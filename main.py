from fastapi import FastAPI, UploadFile, File, HTTPException ,Request,Body
from pydantic import BaseModel
import openai
from io import BytesIO
from typing import Dict
import os
from dotenv import load_dotenv
from generate_summary import generate_summary_gpt4o
import json
import openai
from pydub import AudioSegment
load_dotenv()
# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY 
openai.api_type="openai"

app = FastAPI()

class TextRequest(BaseModel):
    text: str

def mp3_to_wav(mp3_audio: BytesIO) -> BytesIO:
    # MP3 파일을 WAV 파일로 변환
    audio = AudioSegment.from_mp3(mp3_audio)
    wav_audio = BytesIO()
    audio.export(wav_audio, format="wav")
    wav_audio.seek(0)  # Ensure the file pointer is at the beginning
    return wav_audio

def transcribe_audio_whisper(audio_file: BytesIO) -> str:
    # MP3 파일을 WAV 파일로 변환
    wav_audio = mp3_to_wav(audio_file)
    audio_file.seek(0)  # Ensure file pointer is at the beginning
    client = openai.OpenAI()
    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=wav_audio 
)
    return transcription['text']

@app.get("/")
async def read_root():
    return {"message": "Welcome to the GPT-4o and Whisper API!"}

# 요청 데이터 모델
class TextRequest(BaseModel):
    text: str

# 엔드포인트 정의
# 예시:
text="너는 누구니"
@app.post("/text")
async def handle_text():
    summary_text = generate_summary_gpt4o(text, api_key=OPENAI_API_KEY)
    return {"summary": summary_text}
@app.post("/audio")
async def handle_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV and MP3 files are supported.")
    
    audio_content = await file.read()
    audio_file = BytesIO(audio_content)
    
    # Whisper를 사용하여 MP3 또는 WAV 파일의 텍스트를 추출
    transcription_text = transcribe_audio_whisper(audio_file)
    
    # GPT-4o를 사용하여 텍스트 요약 생성
    summary_text = generate_summary_gpt4o(transcription_text)
    
    return {"response": summary_text}
