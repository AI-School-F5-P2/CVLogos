import streamlit as st
from model_loader import load_model
from utils import load_image, run_detection, draw_detections, process_video
import os
import time

def main():
    st.title("OBJECT LOGO DETECTION")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")

    # Configuración de la barra lateral
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left:-300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Selección del modo de la aplicación
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Run on Image', 'Run on Video'])

    if app_mode == 'About App':
        st.markdown('In this project we are using **yoloV8** to do Object Detection (LOGOS) on Images and Videos and we are using **Streamlit** to create a Graphical User Interface.')

    elif app_mode == 'Run on Image':
        run_image_detection()

    # Dentro de app.py
# Modifica la sección "Run on Video" de la siguiente manera:

    elif app_mode == 'Run on Video':
        run_video_detection()

"""
def run_image_detection():
    # Cargar y procesar imágenes
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        model = load_model()  # Asegúrate de que esta función esté correctamente definida en model_loader.py
        detections = run_detection(model, image)
        image_with_detections = draw_detections(image, detections)
        st.image(image_with_detections, use_column_width=True)
        print("Imagen con detecciones generada correctamente")
"""

def run_image_detection():
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        detections, original_size = run_detection(load_model(), image)  # Obtener también el tamaño original
        image_with_detections = draw_detections(image, detections, original_size)  # Pasar el tamaño original
        st.image(image_with_detections, use_column_width=True)
        print("Imagen con detecciones generada correctamente")



def run_video_detection():
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_file is not None:
        st.write("Video cargado correctamente.")
        video_path = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            st.write("Video guardado temporalmente.")

        model = load_model()  # Asegúrate de que esta función esté correctamente definida en model_loader.py
        process_video(video_path, model)
        st.write("Video procesado exitosamente.")

        # Espera para asegurarse de que el archivo de video esté disponible
        time.sleep(5)  # Espera 5 segundos

        processed_video_path = os.path.abspath('runs/output.mp4')  # Ruta absoluta
        if os.path.exists(processed_video_path):
            st.video(processed_video_path)
        else:
            st.write("No se encontró el video procesado.")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
