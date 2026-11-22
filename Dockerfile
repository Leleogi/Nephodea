# Recipe for the server that runs Nephodea on Hugging Face Spaces.
# Hugging Face reads this file, builds the app and starts it.

FROM python:3.10-slim

# Hugging Face runs the app as a normal user (id 1000), not as admin
RUN useradd --create-home --uid 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /home/user/app

# Install the CPU-only PyTorch first: it is much smaller than the default one
RUN pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2

# Then the other packages from requirements.txt.
# The normal OpenCV needs screen libraries a server does not have,
# so it is swapped for the "headless" version made for servers.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --user opencv-python-headless==4.9.0.80

# Finally the app itself and the model
COPY --chown=user current_model.pt .
COPY --chown=user src ./src

# Hugging Face Spaces expects the app on port 7860.
# gunicorn is a production web server; Flask's own server is for testing only.
EXPOSE 7860
CMD ["gunicorn", "--chdir", "src", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "webapp:app"]
