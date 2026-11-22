
import sys
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "current_model.pt"


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    # Load a model
    model = YOLO(str(MODEL_PATH))

    # Run inference on the image
    result = model(sys.argv[1], device="cpu", verbose=False)[0]

    # Print the 5 most likely classes
    probs = result.probs  # Probs object for classification outputs
    for class_id, confidence in zip(probs.top5, probs.top5conf.tolist()):
        print(f"{result.names[class_id]}: {confidence:.1%}")


if __name__ == "__main__":
    main()
