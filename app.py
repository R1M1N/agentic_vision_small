# app_gradio.py
import gradio as gr
import requests
from PIL import Image
import io
import json
from gradio_image_annotation import image_annotator

# API Endpoints remain the same
API_URL_TEXT = "http://127.0.0.1:8000/detect-from-text/"
API_URL_IMAGE = "http://127.0.0.1:8000/detect-from-image-prompt/"
API_URL_DETECT_SEGMENT = "http://127.0.0.1:8000/detect-and-segment/"
API_URL_SEGMENT_POINTS = "http://127.0.0.1:8000/segment-with-points/"
API_URL_SEGMENT_BOX = "http://127.0.0.1:8000/segment-with-box/"
API_URL_SEGMENT_TEXT = "http://127.0.0.1:8000/segment-with-text/"

# --- Handler Functions (no changes needed) ---
def handle_text_detection(image, text_prompt, threshold):
    if image is None: raise gr.Error("Please upload an image.")
    if not text_prompt: raise gr.Error("Please provide a text prompt.")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    data = {'text_prompt': text_prompt, 'threshold': threshold}
    files = {'image_file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
    response = requests.post(API_URL_TEXT, files=files, data=data)
    if response.status_code == 200: return Image.open(io.BytesIO(response.content))
    else: raise gr.Error(f"API Error: {response.text}")

def handle_image_detection(annotated_data, threshold):
    if not annotated_data or not annotated_data.get("image"): raise gr.Error("Please upload an image.")
    if not annotated_data.get("boxes"): raise gr.Error("Please draw a bounding box.")
    image = annotated_data['image']
    box = annotated_data['boxes'][0]
    bbox_coords = (box['xmin'], box['ymin'], box['xmax'], box['ymax'])
    query_image = image.crop(bbox_coords)
    target_byte_arr, query_byte_arr = io.BytesIO(), io.BytesIO()
    image.save(target_byte_arr, format='PNG')
    query_image.save(query_byte_arr, format='PNG')
    data = {'threshold': threshold}
    files = {'target_image_file': ('target.png', target_byte_arr.getvalue()), 'query_image_file': ('query.png', query_byte_arr.getvalue())}
    response = requests.post(API_URL_IMAGE, files=files, data=data)
    if response.status_code == 200: return Image.open(io.BytesIO(response.content))
    else: raise gr.Error(f"API Error: {response.text}")

def handle_detect_and_segment(image, prompt):
    if image is None: raise gr.Error("Please upload an image.")
    if not prompt: raise gr.Error("Please provide a prompt.")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    data = {'prompt': prompt}
    files = {'image_file': ('image.png', img_byte_arr.getvalue())}
    response = requests.post(API_URL_DETECT_SEGMENT, files=files, data=data)
    if response.status_code == 200: return Image.open(io.BytesIO(response.content))
    else: raise gr.Error(f"API Error: {response.text}")

def handle_point_segmentation(image, evt: gr.SelectData):
    points, labels = [evt.index], [1]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    data = {'points': json.dumps(points), 'labels': json.dumps(labels)}
    files = {'image_file': ('image.png', img_byte_arr.getvalue())}
    response = requests.post(API_URL_SEGMENT_POINTS, files=files, data=data)
    if response.status_code == 200: return Image.open(io.BytesIO(response.content))
    else: raise gr.Error(f"API Error: {response.text}")

def handle_box_segmentation(annotated_data):
    if not annotated_data or not annotated_data.get("image"): raise gr.Error("Please upload an image.")
    if not annotated_data.get("boxes"): raise gr.Error("Please draw a bounding box.")
    image = annotated_data['image']
    box = annotated_data['boxes'][0]
    bbox_coords = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    data = {'box': json.dumps(bbox_coords)}
    files = {'image_file': ('image.png', img_byte_arr.getvalue())}
    response = requests.post(API_URL_SEGMENT_BOX, files=files, data=data)
    if response.status_code == 200: return Image.open(io.BytesIO(response.content))
    else: raise gr.Error(f"API Error: {response.text}")

def handle_text_segmentation(image, text_prompt):
    if image is None: raise gr.Error("Please upload an image.")
    if not text_prompt: raise gr.Error("Please provide a text prompt.")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    data = {'text_prompt': text_prompt}
    files = {'image_file': ('image.png', img_byte_arr.getvalue())}
    response = requests.post(API_URL_SEGMENT_TEXT, files=files, data=data)
    if response.status_code == 200: return Image.open(io.BytesIO(response.content))
    else: raise gr.Error(f"API Error: {response.text}")

# --- Build the Redesigned Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Interactive Vision Tasks") as demo:
    gr.Markdown("# Interactive Vision with OWL-ViT and SAM")
    
    with gr.Row():
        # --- Left Column: All Controls and Inputs ---
        with gr.Column(scale=1):
            with gr.Accordion("Advanced Settings", open=False):
                threshold_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.1,
                    label="Confidence Threshold",
                    info="Applies to detection tasks. Lower values find more objects."
                )

            with gr.Tabs():
                # --- Main Tab 1: Detection Tasks ---
                with gr.TabItem("🔎 Detection (OWL-ViT)"):
                    with gr.Tabs():
                        with gr.TabItem("Text Prompt"):
                            gr.Markdown("1. Upload an image.\n2. Enter a text description.\n3. Click 'Detect'.")
                            text_det_input_image = gr.Image(type="pil", label="Input Image")
                            text_det_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g., a cat, a blue car")
                            text_det_btn = gr.Button("Detect with Text", variant="primary")
                        
                        with gr.TabItem("Image Prompt"):
                            gr.Markdown("1. Upload an image and draw a box.\n2. Click 'Detect'.")
                            img_det_annotator = image_annotator(image_type="pil", single_box=True, label="Draw BBox Prompt")
                            img_det_btn = gr.Button("Detect with Image Region", variant="primary")

                # --- Main Tab 2: Segmentation Tasks ---
                with gr.TabItem("🎨 Segmentation (SAM)"):
                    with gr.Tabs():
                        with gr.TabItem("Point Prompt"):
                            gr.Markdown("1. Upload an image.\n2. Click on an object to segment it.")
                            point_seg_input_image = gr.Image(type="pil", label="Click on an object")
                        
                        with gr.TabItem("Box Prompt"):
                            gr.Markdown("1. Upload, draw a box.\n2. Click 'Segment'.")
                            box_seg_annotator = image_annotator(image_type="pil", single_box=True, label="Draw BBox Prompt")
                            box_seg_btn = gr.Button("Segment with Box", variant="primary")

                        with gr.TabItem("Text Prompt"):
                            gr.Markdown("1. Upload an image.\n2. Enter a text description.\n3. Click 'Segment'.")
                            text_seg_input_image = gr.Image(type="pil", label="Input Image")
                            text_seg_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g., the red car")
                            text_seg_btn = gr.Button("Segment with Text", variant="primary")
                
                # --- Main Tab 3: Combined Pipeline ---
                with gr.TabItem("✨ Combined Pipeline"):
                    gr.Markdown("1. Upload image.\n2. Enter a complex prompt.\n3. Click 'Run'.")
                    combined_input_image = gr.Image(type="pil", label="Input Image")
                    combined_prompt = gr.Textbox(label="Complex Prompt", placeholder="e.g., box around the cat and segment the dog")
                    combined_btn = gr.Button("Run Pipeline", variant="primary")

        # --- Right Column: Single Output Display ---
        with gr.Column(scale=1):
            gr.Markdown("## Results")
            output_image = gr.Image(type="pil", label="Output", interactive=False)

    # --- Event Handlers ---
    text_det_btn.click(handle_text_detection, [text_det_input_image, text_det_prompt, threshold_slider], output_image)
    img_det_btn.click(handle_image_detection, [img_det_annotator, threshold_slider], output_image)
    point_seg_input_image.select(handle_point_segmentation, [point_seg_input_image], output_image)
    box_seg_btn.click(handle_box_segmentation, [box_seg_annotator], output_image)
    text_seg_btn.click(handle_text_segmentation, [text_seg_input_image, text_seg_prompt], output_image)
    combined_btn.click(handle_detect_and_segment, [combined_input_image, combined_prompt], output_image)

if __name__ == "__main__":
    demo.launch()

