from config.core.config import TempVideoManager 
from pipeline import format_report
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from backend.src.llm.manager import LLMManager
from backend.src.llm.prompt_builder import build_prompt


app = FastAPI()

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow origin
    allow_credentials=True,  # Allow cookie/credential
    allow_methods=["*"],  # Allow http action
    allow_headers=["*"],  # Allow headers
)


@app.post("/upload-video")
async def upload_video(
    # Define video parameters
    video: UploadFile = File(),
    # Define action
    action: str = Form(),
    
) -> dict:
    # Try catch the CV Layer Errors
    try:
        report = format_report(video)
        prompt = build_prompt(report=report, action=action)
    except Exception as e:
        # This instantly triggers React's catch block
        print(f"CV Layer Error {e}")
        raise HTTPException(status_code=422, detail="Video processing failed.")
    # Try catch the LLM Errors
    try:
        manager = LLMManager()
        feedback = await manager.manage_model(prompt=prompt)

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
