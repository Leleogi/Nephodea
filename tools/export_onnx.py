"""Convert current_model.pt into files the browser version can use.

Run from the project folder:  python tools/export_onnx.py
Needs:  pip install onnx==1.16.1

It writes two files into space/:
- nephodea.onnx  the model in ONNX format, which runs in a web browser
- labels.json    the 12 class names, in the model's own order
"""

import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
SPACE = ROOT / "space"

model = YOLO(str(ROOT / "current_model.pt"))

# Ultralytics writes the .onnx file next to the .pt file; move it into space/
onnx_file = model.export(format="onnx", imgsz=640, opset=12, simplify=False)
shutil.move(onnx_file, SPACE / "nephodea.onnx")

labels = [model.names[i] for i in range(len(model.names))]
(SPACE / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")

print("Wrote space/nephodea.onnx and space/labels.json")
