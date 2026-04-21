from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

origins=[
    'http://localhost:3000',
    'http://127.0.0.1:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # Allow origin
    allow_credentials=True, # Allow cookie/credential
    allow_methods=['*'],    # Allow http action
    allow_headers=['*']     # Allow headers
)

@app.post("/upload-video")
async def upload_video(
    # Define video parameters
    video: UploadFile = File(),
    
    # Define action
    action:str= Form()
    
):
    print(f"Received File {video.filename}")
    print(f"Action type: {action}")
    
    return{
        'status': 'success',
        'filename': video.filename,
        'action_received': action,
        'message': "Video uploaded and processed"
    }