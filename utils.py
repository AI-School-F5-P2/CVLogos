import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.utils import draw_bounding_boxes
import numpy as np
import matplotlib.pyplot as plt
from model_loader import load_model
import os
from pytube import YouTube
from collections import Counter

def load_image(uploaded_file):
    """
    Carga un archivo de imagen subido y lo convierte en un formato adecuado para el procesamiento.
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Convertir de BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image, target_size=(640, 640)):
    print(f"Imagen original: {image.shape}, tipo: {image.dtype}")  # Información de la imagen original
    image = cv2.resize(image, target_size)
    print(f"Imagen preprocesada: {image.shape}, tipo: {image.dtype}")  # Información de la imagen preprocesada
    return image


def run_detection(model, image):
    original_size = image.shape[:2]  # Guardar dimensiones originales
    preprocessed_image = preprocess_image(image)
    results = model(preprocessed_image)

    detections = []
    if len(results) > 0:
        # Suponiendo que el primer elemento en 'results' tiene la información de detección
        detection_data = results[0]

        for box in detection_data.boxes:
            # Extrayendo las coordenadas del cuadro delimitador de 'xyxy'
            x1, y1, x2, y2 = box.xyxy[0]

            # Obteniendo la confianza y el ID de la clase
            conf = box.conf[0].item()
            cls_id = box.cls[0].item()
            cls_name = detection_data.names[int(cls_id)]  # Convertir el ID de la clase a su nombre

            detections.append((x1, y1, x2, y2, conf, cls_name))

    print("Processed detections:", detections)
    return detections, original_size

def draw_detections(image, detections, original_size):
    original_height, original_width = original_size
    scale_y, scale_x = original_height / 640, original_width / 640

    for x1, y1, x2, y2, conf, cls in detections:
        # Redondear las coordenadas y convertirlas a enteros
        #x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
        x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

# Convertir a enteros
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        label = f'{cls} {conf:.2f}'
        # Dibuja el cuadro delimitador y la etiqueta en la imagen
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image


"""
def draw_detections(image, detections):
    # Obtiene las dimensiones de la imagen
    image_height, image_width, _ = image.shape

    # Dibuja los cuadros delimitadores y etiquetas de las detecciones en la imagen.
    for detection in detections:
        # Asumiendo que cada 'detection' es un tensor con la forma [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = int(detection[0].item()), int(detection[1].item()), int(detection[2].item()), int(detection[3].item()), detection[4], detection[5]

        # Calcula las coordenadas para centrar el cuadro en el objeto detectado
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        new_x1 = max(0, center_x - width // 2)
        new_x2 = min(image_width - 1, center_x + width // 2)
        new_y1 = max(0, center_y - height // 2)
        new_y2 = min(image_height - 1, center_y + height // 2)

        label = f'{cls} {conf:.2f}'  # Ajusta esto según tus etiquetas de clase

        # Dibuja el cuadro delimitador y la etiqueta en la imagen
        cv2.rectangle(image, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
        cv2.putText(image, label, (new_x1, new_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Dibujando cuadro: {new_x1}, {new_y1}, {new_x2}, {new_y2}, clase: {cls}, confianza: {conf:.2f}")

    return image"""



def process_video(video_file, model):
    output_folder = 'runs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, 'output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    
    all_detections = []  # Lista para almacenar detecciones de todos los frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, original_size = run_detection(model, frame)
        frame_with_detections = draw_detections(frame, detections, original_size)
        all_detections.extend(detections)  # Acumulando detecciones

        if frame_with_detections is not None:
            out.write(frame_with_detections)

    cap.release()
    out.release()
    print(f"Video guardado en {output_path}")
    return all_detections, output_path


def download_video_from_url(youtube_url):
    try:
        yt = YouTube(youtube_url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        return video.download()
    except Exception as e:
        print(f"Error al descargar el video: {e}")
        return None
    

def calculate_bbox_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)



def generate_report(detections):
    report_data = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_name = det
        bbox_area = (x2 - x1) * (y2 - y1)
        report_data.append({
            'Class': cls_name,
            'Confidence': f"{conf:.2f}",
            'BBox Coords': f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]",
            'BBox Area': f"{bbox_area:.2f}"
        })
    return report_data



def generate_summary(detections):
    total_detections = len(detections)
    class_counter = Counter([det[5] for det in detections])  # Contando detecciones por clase
    summary = f"Total Detections: {total_detections}\n"
    summary += "\n".join([f"{cls}: {count} detections" for cls, count in class_counter.items()])
    return summary
