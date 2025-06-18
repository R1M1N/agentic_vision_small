# api.py
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, Response
from PIL import Image
import io
import json # <--- ADD THIS LINE
import torch
import traceback
from core.segmentor import Segmentor
from core.detector import OWLViTDetector
from ui.visualizer import ResultsVisualizer
from core.combined_pipeline import OwlViT_SAM_Pipeline
combined_pipeline = OwlViT_SAM_Pipeline()


app = FastAPI(title="Agent Vision Small")
segmentor = Segmentor()
detector = OWLViTDetector()
visualizer = ResultsVisualizer()

# @app.exception_handler(Exception)
# async def generic_exception_handler(request, exc):
#     traceback.print_exc()
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "An internal server error occurred.", "error": str(exc)},
#     )

@app.post("/detect-and-segment/")
async def detect_and_segment(
    prompt: str = Form(...),
    image_file: UploadFile = File(...)
):
    """API endpoint for combined OWL-ViT detection and SAM segmentation."""
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Run the combined pipeline
    result_image, _, _ = combined_pipeline.run(image, prompt)
    
    # Convert result image to bytes to send back to the client
    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    
    return Response(content=buffered.getvalue(), media_type="image/png")

@app.post("/detect-from-text/")
async def detect_from_text(
    text_prompt: str = Form(...),
    image_file: UploadFile = File(...),
    threshold: float = Form(...)  # Add threshold parameter
):
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Use the threshold from the form
    results = detector.detect_from_text(image, text_prompt, threshold=threshold)
    
    result_image = visualizer.draw_detections(image.copy(), results)
    
    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    
    return Response(content=buffered.getvalue(), media_type="image/png")


@app.post("/detect-from-image-prompt/")
async def detect_from_image_prompt(
    target_image_file: UploadFile = File(...),
    query_image_file: UploadFile = File(...),
    threshold: float = Form(...)  # Add threshold parameter
):
    target_image_bytes = await target_image_file.read()
    target_image = Image.open(io.BytesIO(target_image_bytes)).convert("RGB")

    query_image_bytes = await query_image_file.read()
    query_image = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")

    # Use the threshold from the form
    results = detector.detect_similar_objects(target_image, query_image, threshold=threshold)
    
    result_image = visualizer.draw_detections(target_image.copy(), results)

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    
    return Response(content=buffered.getvalue(), media_type="image/png")
@app.post("/segment-with-points/")
async def segment_with_points_endpoint(
    points: str = Form(...), # JSON string of points
    labels: str = Form(...), # JSON string of labels
    image_file: UploadFile = File(...)
):
    image = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
    result_image = segmentor.segment_with_points(image, json.loads(points), json.loads(labels))
    
    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    return Response(content=buffered.getvalue(), media_type="image/png")

@app.post("/segment-with-box/")
async def segment_with_box_endpoint(
    box: str = Form(...), # JSON string of the box
    image_file: UploadFile = File(...)
):
    image = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
    result_image = segmentor.segment_with_box(image, json.loads(box))

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    return Response(content=buffered.getvalue(), media_type="image/png")

@app.post("/segment-with-text/")
async def segment_with_text_endpoint(
    text_prompt: str = Form(...),
    image_file: UploadFile = File(...)
):
    image = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
    result_image = segmentor.segment_with_text(image, text_prompt)

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    return Response(content=buffered.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
