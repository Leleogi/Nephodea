import base64
import io
import os
from pathlib import Path
from threading import Lock

from flask import Flask, render_template, request
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / "current_model.pt"
MAX_UPLOAD_MB = 10
MAX_PIXELS = 60_000_000  # protects the server from huge images that fill the memory
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "HEIF"}  # MPO is a JPEG type many phones produce
UNSURE_BELOW = 0.5  # below this probability we tell the user the model is unsure
PREVIEW_SIZE = 1200  # longest side of the image shown on the page, in pixels

# Lets Pillow open iPhone HEIC photos
register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Load the model once at startup, not per request
model = YOLO(str(MODEL_PATH))
model_lock = Lock()  # the model handles one photo at a time


class PhotoError(Exception):
    """A problem with the upload. The message is shown to the user."""


def load_photo(upload):
    """Check the uploaded file and return it as an upright image, in memory only."""
    bad_file = "This file could not be read as a photo. Please choose a JPEG, PNG or HEIC image."
    try:
        # Opening the file checks its real content, whatever its name says
        photo = Image.open(upload.stream)
        if photo.format not in ALLOWED_FORMATS:
            raise PhotoError(bad_file)
        if photo.width * photo.height > MAX_PIXELS:
            raise PhotoError("This photo has too many pixels. Please choose a smaller one.")
        # Turn the photo upright using its EXIF rotation information
        return ImageOps.exif_transpose(photo).convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise PhotoError(bad_file)


def classify(photo):
    """Return the 3 most likely classes as a list of (class name, probability)."""
    with model_lock:
        probs = model(photo, device="cpu", verbose=False)[0].probs
    top_classes = zip(probs.top5[:3], probs.top5conf[:3].tolist())
    return [(model.names[class_id], confidence) for class_id, confidence in top_classes]


def display_name(class_name):
    """Make a class name readable, e.g. mixed_clouds becomes Mixed clouds."""
    if class_name == "not_cloud":
        return "No clouds"
    return class_name.replace("_", " ").capitalize()


def make_preview(photo):
    """Return a smaller copy to show on the page. It contains no GPS or other photo data."""
    preview = photo.copy()
    preview.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_result(photo):
    """Put together everything the page needs to show the result."""
    top_classes = classify(photo)
    results = [
        {"name": display_name(name), "percent": f"{confidence * 100:.1f} %"}
        for name, confidence in top_classes
    ]
    return {
        "top": results[0],
        "others": results[1:],
        "no_clouds": top_classes[0][0] == "not_cloud",
        "unsure": top_classes[0][1] < UNSURE_BELOW,
        "preview": make_preview(photo),
    }


# routes
@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template("index.html", max_mb=MAX_UPLOAD_MB)

    upload = request.files.get("photo")
    if upload is None or upload.filename == "":
        return show_error("Please choose a photo first.", 400)

    try:
        photo = load_photo(upload)
    except PhotoError as error:
        return show_error(str(error), 400)

    return render_template("index.html", max_mb=MAX_UPLOAD_MB, result=build_result(photo))


@app.route("/about")
def about_page():
    return "Test Version of the Cloud Image Classifier CloudAI by Leandro Gimmi, Michael Herrmann, Capucine Lechartre and Melina Abeling"


def show_error(message, status):
    return render_template("index.html", max_mb=MAX_UPLOAD_MB, error=message), status


# Flask calls this when an upload is over MAX_CONTENT_LENGTH
@app.errorhandler(413)
def file_too_large(_error):
    return show_error(f"This file is too large. Please choose a photo under {MAX_UPLOAD_MB} MB.", 413)


if __name__ == "__main__":
    # Debug mode stays off unless you ask for it, because it is unsafe on a public server.
    # Port 5001 because macOS often uses 5000 for AirPlay.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5001)
