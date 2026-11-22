# Nephodea

Nephodea is a small web app that recognises cloud types in sky photos. You upload a photo and it shows the most likely cloud type with its probability, plus the next most likely ones. It can also tell when a photo probably shows no clouds.

This is a learning project and still a test version. The model makes mistakes, so treat its answers as a guess, not a weather forecast.

## How it works

The model is a small image classifier ([Ultralytics YOLOv8n-cls](https://docs.ultralytics.com/tasks/classify/), about 3 MB). It looks at the whole photo and gives one probability for each of 12 classes:

altocumulus, altostratus, cirrocumulus, cirrostratus, cirrus, cumulonimbus, cumulus, mixed clouds, nimbostratus, not cloud, stratocumulus, stratus

On its validation set, the correct class was the top answer for 73 % of the photos, and among the top 5 answers for 96 %.

## Cloud regions

The live page also looks at clouds one by one, in three steps:

1. **Find the sky.** A small sky segmentation model (U-2-Net, about 4.5 MB) marks which pixels of the photo are sky.
2. **Tell clouds from blue sky.** Inside the sky, a colour rule decides for each pixel: clear sky scatters much more blue light than red, while clouds look white or grey. A pixel counts as cloud when (blue − red) / (blue + red) is 0.05 or less.
3. **Classify each cloud region.** Cloud pixels that touch are grouped into regions. The biggest ones (up to 4) are cut out as squares and classified on their own.

The colour rule is simple, so a hazy sky near the horizon, a sunset or a night sky can be mistaken for cloud.

On the live page the model runs in your browser, so photos are never uploaded. The Flask version keeps uploaded photos in memory only while it analyses them and never saves them.

## Run it on your own computer

You need Python 3.10.

```bash
python3 -m venv venv310
source venv310/bin/activate
pip install -r requirements.txt
```

Classify one photo from the command line:

```bash
python predict.py path/to/photo.jpg
```

This prints the 5 most likely cloud types.

Start the web app, then open http://127.0.0.1:5001 in your browser:

```bash
python src/webapp.py
```

## How it is hosted

The live page is a free **Static** Hugging Face Space: https://huggingface.co/spaces/Leleogi/nephodea

There is no server. The `space/` folder holds a plain web page, the model in ONNX format (`nephodea.onnx`) and [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (MIT licence), which runs the model inside the visitor's browser. Photos never leave the visitor's device.

Every merge into `main` that changes `space/` uploads it to the Space automatically (`.github/workflows/deploy-space.yml`).

To try the browser version on your own computer:

```bash
cd space
python -m http.server 8000
```

Then open http://127.0.0.1:8000.

After changing `current_model.pt`, rebuild the browser model with `python tools/export_onnx.py` (needs `pip install onnx==1.16.1`).

The sky model `space/sky.onnx` was made with `python tools/convert_sky_model.py` (needs `pip install onnx==1.16.1 numpy`). The author only publishes it in the ncnn format for phones, so the script downloads those files and rewrites the same network as ONNX, without changing its weights.

The `Dockerfile` runs the Flask version on a server instead (port 7860). It is kept for hosts that run Docker, for example a paid Hugging Face Space or Google Cloud Run.

## Project files

| File | What it does |
| --- | --- |
| `src/webapp.py` | The web app (Flask) |
| `src/templates/index.html` | The upload and result page |
| `predict.py` | Classifies one photo from the command line |
| `current_model.pt` | The trained model |
| `requirements.txt` | The Python packages, with pinned versions |
| `space/` | The browser version that runs on Hugging Face |
| `tools/export_onnx.py` | Converts the model for the browser |
| `tools/convert_sky_model.py` | Converts the sky model for the browser |
| `Dockerfile` | The recipe for running the Flask version on a server |

## Credits

Thanks to Michael Herrmann, Capucine Lechartre and Melina Abeling, who helped train this model.

The original code was created at MeteoSwiss (Federal Office of Meteorology and Climatology) and is used under the MIT licence, see `LICENSE`.

The sky segmentation model (`space/sky.onnx`) is the small U-2-Net sky model from [Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing) by xiongzhu666, converted to ONNX, MIT licence.

The cloud colour rule (normalised red-blue ratio, threshold 0.05) comes from [Cloud detection in sky images](https://github.com/yuhao-nie/Cloud-dection-in-sky-images) by Yuhao Nie, MIT licence.

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0.
