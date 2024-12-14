from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import subprocess
from insert_log_info import insert_log
from inferens_transformer import infer
from datetime import datetime
import uvicorn

# 애플리케이션 인스턴스 생성
app = FastAPI()

# CORS 설정
origins = [
    "http://0.0.0.0:8001",  # 서버와 동일한 도메인
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 경로 설정
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
PROCESSED_DIR = os.path.join(STATIC_DIR, "processed")

# 디렉토리 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# StaticFiles와 Jinja2Templates 설정
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/")
def read_root(request: Request):
    """메인 페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """동영상 처리"""
    try:
        # 파일 저장 경로 설정
        unique_filename = file.filename
        upload_path = os.path.join(UPLOAD_DIR, unique_filename)
        processed_filename = f"processed_{unique_filename}"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)

        # 업로드된 파일 저장
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 단순 복사 후 AI 처리 호출
        shutil.copy(upload_path, processed_path)
        event_logs = infer(processed_path)

        # ffmpeg 명령어 실행 (동영상 포맷 변환)
        output_path = os.path.join(PROCESSED_DIR, "overlay_video_h264.mp4")
        ffmpeg_command = [
            "ffmpeg",
            "-i", processed_path,  # 입력 파일 경로
            "-c:v", "libx264",     # 비디오 코덱
            "-preset", "slow",     # 인코딩 속도
            "-crf", "23",          # 비디오 품질 설정
            output_path            # 출력 파일 경로
        ]
        subprocess.run(ffmpeg_command, check=True)

        # 처리된 동영상 URL 반환
        processed_video_url = f"/static/processed/overlay_video_h264.mp4"

        return JSONResponse({
            "processed_video_url": processed_video_url,
            "event_logs": event_logs
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# 서버 실행
if __name__ == "__main__":
    # uvicorn.run(app, host= "localhost",port = 5001)
    # 접속 url
    # http://112.175.29.231:8011/
    insert_log('web server start', 'event', 'file_url or frame count')
    #uvicorn.run(app, host="0.0.0.0", port=8001)
    uvicorn.run(app, host="192.168.33.11", port=8001)
