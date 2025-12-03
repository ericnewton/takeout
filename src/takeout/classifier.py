from ultralytics import YOLO
import numpy as np
import cv2
MODEL = YOLO('yolo11n-cls.pt', verbose=False)
THRESHOLD = 0.5

def classify(img: np.ndarray) -> list[str]:
    result = []
    results = MODEL(img, verbose=False)
    for classification in results:
        for summary in classification.summary():
            if summary['confidence'] > THRESHOLD:
                name = summary['name']
                result.append(name)
                if '_' in name:
                    result += name.split('_')
    return result
