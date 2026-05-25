import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Xử Lý Ảnh AI Backend")

# Phục vụ thư mục frontend (HTML/CSS/JS)
if not os.path.exists("frontend"):
    os.makedirs("frontend")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

class PipelineRequest(BaseModel):
    image: str

@app.post("/api/pipeline")
async def pipeline_endpoint(req: PipelineRequest):
    if not req.image:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ảnh.")
    
    try:
        from pipeline_utils import process_detection_and_pipeline
        
        if "," in req.image:
            base64_data = req.image.split(",")[1]
        else:
            base64_data = req.image
            
        image_bytes = base64.b64decode(base64_data)
        result = process_detection_and_pipeline(image_bytes)
        return result
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
