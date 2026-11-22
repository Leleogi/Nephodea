# Nephodea

Nephodea is a small web app that recognises cloud types in sky photos. You upload a photo and it shows the most likely cloud type with its probability, plus the next most likely ones. It can also tell when a photo probably shows no clouds.

This is a learning project and still a test version. The model makes mistakes, so treat its answers as a guess, not a weather forecast.

## How it works

The model is a small image classifier ([Ultralytics YOLOv8n-cls](https://docs.ultralytics.com/tasks/classify/), about 3 MB). It looks at the whole photo and gives one probability for each of 12 classes:

altocumulus, altostratus, cirrocumulus, cirrostratus, cirrus, cumulonimbus, cumulus, mixed clouds, nimbostratus, not cloud, stratocumulus, stratus

On its validation set, the correct class was the top answer for 73 % of the photos, and among the top 5 answers for 96 %.

Uploaded photos are only kept in memory while they are analysed. They are never saved.

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

Start the web app, then open http://127.0.0.1:5001 in your browser:

```bash
python src/webapp.py
```

## Project files

| File | What it does |
| --- | --- |
| `src/webapp.py` | The web app (Flask) |
| `src/templates/index.html` | The upload and result page |
| `predict.py` | Classifies one photo from the command line |
| `current_model.pt` | The trained model |
| `requirements.txt` | The Python packages, with pinned versions |

## Credits

Thanks to Michael Herrmann, Capucine Lechartre and Melina Abeling, who helped train this model.

The original code was created at MeteoSwiss (Federal Office of Meteorology and Climatology) and is used under the MIT licence, see `LICENSE`.

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0.
