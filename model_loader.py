from ultralytics import YOLO

def load_model(model_path='models/best.pt'):
    # Carga el modelo usando ultralytics YOLO
    model = YOLO(model_path)
    return model
