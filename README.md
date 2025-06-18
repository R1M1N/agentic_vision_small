# Interactive Vision with OWL-ViT and SAM

This project is a powerful, interactive web application that combines state-of-the-art computer vision models to perform a variety of detection and segmentation tasks. Users can interact with the models through an intuitive Gradio web interface, which communicates with a robust FastAPI backend.

## Overview

The application provides a unified platform for:
*   **Open-Vocabulary Object Detection**: Using Google's **OWL-ViT**, you can detect objects based on text descriptions or image-based prompts.
*   **High-Precision Segmentation**: Using Meta AI's **Segment Anything Model (SAM)**, you can generate detailed segmentation masks for objects by providing points, bounding boxes, or text prompts.
*   **Combined Intelligent Pipelines**: An advanced workflow that uses OWL-ViT to first detect objects and then passes the results to SAM for segmentation, enabling complex, multi-step instructions.

## Features

*   **Interactive Web UI**: A user-friendly interface built with Gradio for easy interaction.
*   **Multiple Task Modes**:
    *   **Detection (OWL-ViT)**:
        *   **Text Prompt**: Find objects using natural language descriptions (e.g., "a red car").
        *   **Image Prompt**: Draw a box around an object to find similar ones.
    *   **Segmentation (SAM)**:
        *   **Point Prompt**: Click on an object to instantly segment it.
        *   **Box Prompt**: Draw a box to get a precise mask of the enclosed object.
        *   **Text Prompt**: Describe an object to detect and segment it automatically.
    *   **Advanced Combined Pipeline**:
        *   Process complex commands like "box around the cat and segment the dog" in a single step.
*   **Adjustable Confidence**: An interactive slider to fine-tune the detection threshold for optimal results.
*   **Scalable API Backend**: A robust FastAPI server that handles model inference and can be scaled for production use.

## Architecture

The project is built on a modern, decoupled architecture:
*   **FastAPI Backend (`api.py`)**: A high-performance Python server that loads the models (OWL-ViT, SAM) into memory and exposes their functionality through a set of REST API endpoints. It handles all the heavy computation and model inference.
*   **Gradio Frontend (`app_gradio.py`)**: A reactive web interface that provides all the user-facing controls. It does not perform any model inference itself; instead, it sends user inputs (images, prompts, coordinates) to the FastAPI backend and displays the results.

## Project Structure

```
.
├── api.py                  # FastAPI backend server
├── app_gradio.py           # Gradio frontend UI
├── requirements.txt        # Project dependencies
│
├── core/
│   ├── detector.py           # OWL-ViT detection logic
│   ├── segmentor.py          # SAM segmentation logic
│   └── combined_pipeline.py  # OWL-ViT + SAM pipeline logic
│
└── sam_vit_h_4b8939.pth      # SAM model checkpoint
```

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository**
```bash
git clone https://github.com/R1M1N/agentic_vision_small.git
cd agentic_vision_small
```

**2. Create a Python Environment**
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
Install all required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Install Segment Anything Model**
The SAM library is installed directly from its source repository for the latest updates.
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**5. Download Model Checkpoints**
You need to download the pre-trained model weights for SAM.
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Place the downloaded `sam_vit_h_4b8939.pth` file in the root directory of your project. The OWL-ViT models will be downloaded automatically by the `transformers` library on first run.

## How to Run the Application

The application requires two separate terminal sessions: one for the backend and one for the frontend.

**Terminal 1: Start the FastAPI Backend**
Navigate to the project's root directory and run the Uvicorn server.
```bash
uvicorn api:app --reload
```
The server will start, typically at `http://127.0.0.1:8000`. Keep this terminal running.

**Terminal 2: Start the Gradio Frontend**
Open a new terminal, navigate to the same project directory, and run the Gradio app.
```bash
python app_gradio.py
```
This will launch the user interface and provide a local URL, usually `http://127.0.0.1:7860`. Open this URL in your web browser to use the application.

## How to Use the Application

The UI is organized into logical tabs for different tasks:
*   **Detection (OWL-ViT)**: Use this tab to find objects. You can either provide a text description or upload an image and draw a bounding box to find similar objects.
*   **Segmentation (SAM)**: Use this tab to generate masks. You can prompt SAM by clicking on an object (Point Prompt), drawing a box around it (Box Prompt), or describing it (Text Prompt).
*   **Combined Pipeline**: Use this tab for advanced, multi-step instructions that involve both detection and segmentation.

The **Confidence Threshold** slider in the "Advanced Settings" accordion applies to all detection tasks, allowing you to control the sensitivity of the model.

## API Endpoints

The FastAPI server exposes the following endpoints for programmatic access:

| Method | Endpoint                      | Description                                    |
| :----- | :---------------------------- | :--------------------------------------------- |
| `POST` | `/detect-from-text/`          | Detects objects from a text prompt.            |
| `POST` | `/detect-from-image-prompt/`  | Detects objects using an image crop as a prompt. |
| `POST` | `/segment-with-points/`       | Segments an object from point coordinates.     |
| `POST` | `/segment-with-box/`          | Segments an object from a bounding box.        |
| `POST` | `/segment-with-text/`         | Segments an object from a text prompt.         |
| `POST` | `/detect-and-segment/`        | Runs the combined detection/segmentation pipeline. |