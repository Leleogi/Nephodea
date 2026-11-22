# Nephodea: Cloud Type Classifier (web app)

## What this project is
A web service where people upload a photo of the sky. A pretrained YOLOv8 model analyzes the image and tells the user which cloud type it shows (with confidence), e.g. cirrus, cumulus, stratus.

Goal: a simple, public-facing tool, web first. A native iOS app is an optional later step.

This started as a 2024 hackathon team project (then called CloudAI). Training was done earlier on a cluster; this repository only serves the finished model.

## Model (verified from the files)
- Framework: Ultralytics YOLOv8n **classification** (`yolov8n-cls`), pinned in `requirements.txt` as `ultralytics==8.2.1`
- Task: image **classification**, not object detection. The model gives one label for the whole image plus probabilities. There are NO bounding boxes.
- Weights: `current_model.pt` (about 3 MB, trained model). Validation accuracy from training: top-1 73 %, top-5 96 %.
- Input size: 640 (used in training)
- Classes (12, in this order, index 0 to 11):
  0 altocumulus, 1 altostratus, 2 cirrocumulus, 3 cirrostratus, 4 cirrus, 5 cumulonimbus, 6 cumulus, 7 mixed_clouds, 8 nimbostratus, 9 not_cloud, 10 stratocumulus, 11 stratus
- Read the class names from the model itself (`model.names`) instead of hardcoding them.
- "not_cloud" means the photo probably does not show clouds. Show a friendly message in that case.
- Showing the top 5 classes with probabilities is the best way to answer "which cloud(s) can be seen", since the model returns one probability per class.
- Do NOT retrain or modify the weights unless I ask.
- The original training data, cluster scripts and project template are NOT in this repository. They stay only in the owner's local folder.

## What already exists
- `src/webapp.py`: Flask app. Loads the model once at startup, checks uploads, keeps photos in memory only and never writes them to disk. Runs on port 5001 because macOS often uses 5000 for AirPlay. Uses Flask's built-in server, which is for local use only.
- `src/templates/index.html`: the upload and result page (no outside CSS or JavaScript).
- `predict.py`: command-line test script. Loads `current_model.pt` and prints the top 5 classes for one image. Works.
- `venv310/`: the environment to use (Python 3.10, Apple Silicon, git-ignored). It depends on `python310/`, which holds the Python itself (also git-ignored). Do not delete `python310/`.
- `venv/`: old, unused Python 3.9 Intel environment with no project packages (git-ignored). Do not use it: OpenCV tries to compile from source there.
- `test_images/`: the owner's test photos (git-ignored, because they contain GPS locations).

## Plan
1. Fix `predict.py` so it loads `current_model.pt` and prints the top 3 classes for one test image.
2. Fix `src/webapp.py` (keep Flask, it is already there) to use that model, show top 3 results with percentages, and delete uploads after analysis.
3. Add upload checks: max file size, allowed formats (JPEG, PNG, HEIC), fix EXIF rotation, resize.
4. Add `requirements.txt` for the web app only (small, pinned) and a Dockerfile. Prefer the CPU version of PyTorch to keep the image small.
5. Deploy: Hugging Face Spaces for a quick demo, or Google Cloud Run (EU region) for my own URL.
6. Later (optional): a JSON API (`POST /predict`) for an iOS app, or an on-device Core ML export.

## Working rules
- Work in small steps and explain what each change does, in plain language.
- Show me the plan before big changes or new dependencies.
- Keep the code simple and readable. I am not a professional developer.
- Pin dependency versions.
- Do not store uploaded images permanently. Process them and delete them.
- Never commit secrets or API keys. Use environment variables.
- Ask me before deleting files, running anything that costs money (cloud deployments), or installing system-wide software.

## Commands (to verify as we go)
- Activate environment: `source venv310/bin/activate`
- Install: `pip install -r requirements.txt`
- Recreate Apple Silicon Python 3.10 if `python310/` is missing: `CONDA_SUBDIR=osx-arm64 ~/opt/anaconda3/bin/conda create -y -p "$PWD/python310" --override-channels -c conda-forge "python=3.10" pip`, then `python310/bin/python -m venv venv310`
- Test the model on a photo: `python predict.py test_images/IMG_3964.jpeg`
- Run the web app locally: `python src/webapp.py`, then open http://127.0.0.1:5001 (add `FLASK_DEBUG=1` in front for debug mode, on your own computer only)

## Current status
- [x] Planning (web first, iOS later)
- [x] Model found and checked: `current_model.pt`, YOLOv8n-cls, 12 classes
- [x] Working local test script (`predict.py`)
- [x] `requirements.txt` for `predict.py` (still needs the web app's packages, and CPU-only PyTorch for Docker)
- [x] Project prepared for Git (not committed yet)
- [x] Fixed Flask web app
- [x] Upload checks and privacy (photos stay in memory only)
- [x] Dockerfile for deployment
- [x] Deploy (free Static Hugging Face Space, model runs in the browser)
- [x] Cloud regions on the live page: sky model (`space/sky.onnx`, MIT, from xiongzhu666) + red/blue colour rule (MIT, Yuhao Nie) + classifier per region. The Flask version does not have this yet.
