
from config.core.config import TempVideoManager
from pipeline import format_report
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from src.llm.manager import LLMManager
from src.llm.prompt_builder import build_prompt
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()


origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(',')


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow origin
    allow_credentials=True,  # Allow cookie/credential
    allow_methods=["*"],  # Allow http action
    allow_headers=["*"],  # Allow headers
)


@app.post("/upload-video")
async def upload_video(
    # Define language
    language: str = Form(default = 'en'),
    # Define video parameters
    video: UploadFile = File(),
    # Define action
    action: str = Form(),
    
) -> dict:
    # -- Block 1 -- CV Layer
    try:       
        # Download the video from the cache into the disk
        # CV/Mediapipe needs to read the file from the disk
        with TempVideoManager(video.file) as temp_video_path:
            report = format_report(temp_video_path)
            print(report)
            prompt = build_prompt(report=report, action=action,language = language)
    except Exception as e:
        # This instantly triggers React's catch block
        print(f"CV Layer Error {e}")
        raise HTTPException(status_code=422, detail="Video processing failed.")
    # -- Block 2 -- LLM Layer
    try:
        manager = LLMManager()
        feedback = await manager.manage_model(prompt=prompt)
        print(feedback)
    
    except Exception as e:
        # This instantly triggers React's catch block
        print(f"LLM Layers Error {e}")
        raise HTTPException(
            status_code=503, detail="AI generation failed. All models are down."
        )
    return {
        "status": "processed", 
        "llm_feedback": feedback
            }
