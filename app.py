import streamlit as st
from model_loader import load_model
from utils import load_image, run_detection, draw_detections, process_video, generate_summary, clean_filename
import os
import time
from utils import download_video_from_url
from utils import load_image, run_detection, draw_detections, process_video, generate_report
import pandas as pd
import subprocess


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
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Run on Image', 'Run on Video', 'Run on Video URL'])

    if app_mode == 'About App':
        st.markdown('In this project we are using **yoloV8** to do Object Detection (LOGOS) on Images and Videos and we are using **Streamlit** to create a Graphical User Interface.')

    elif app_mode == 'Run on Image':
        run_image_detection()

    # Dentro de app.py
# Modifica la sección "Run on Video" de la siguiente manera:

    elif app_mode == 'Run on Video':
        run_video_detection()

    elif app_mode == 'Run on Video URL':
            run_video_url_detection()


def run_image_detection():
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        model = load_model()
        detections, original_size = run_detection(model, image)
        image_with_detections = draw_detections(image, detections, original_size)
        st.image(image_with_detections, use_column_width=True)

        # Generar informe y mostrarlo como una tabla
        report_data = generate_report(detections)
        report_df = pd.DataFrame(report_data)
        st.table(report_df)

        # Generar y mostrar el resumen
        summary = generate_summary(detections)
        st.markdown("### Summary")
        st.markdown(summary)

        # Mostrar un gráfico de barras para las detecciones por clase
        if 'Class' in report_df.columns:
            class_counts = report_df['Class'].value_counts()
            st.bar_chart(class_counts)


        # Opcional: Mostrar un gráfico de barras para las detecciones por clase
        #class_counts = report_df['Class'].value_counts()
        #st.bar_chart(class_counts)



def run_video_detection():
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_file is not None:
        video_path = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Your video is processing. Please, Wait.")

        model = load_model()
        #detections, processed_video_path = process_video(video_path, model)
        
        output_filename = os.path.basename(video_path)  # Nombre del archivo de salida igual al de entrada
        detections, processed_video_path = process_video(video_path, model, output_filename)



        # Aquí se asume que 'process_video' ahora devuelve detecciones junto con el path del video procesado
        if os.path.exists(processed_video_path):
            st.video(processed_video_path)

            # Generar y mostrar informe y resumen
            report_data = generate_report(detections)
            report_df = pd.DataFrame(report_data)
            st.table(report_df)

            summary = generate_summary(detections)
            st.markdown("### Summary")
            st.markdown(summary)

            # Gráfico de barras (opcional)
            if 'Class' in report_df.columns:
                class_counts = report_df['Class'].value_counts()
                st.bar_chart(class_counts)


def run_video_url_detection():
    video_url = st.text_input("Enter the video URL")
    if video_url:
        video_path, video_title = download_video_from_url(video_url)
        if video_path and os.path.exists(video_path):
            # El archivo de video existe y se descargó correctamente

            output_folder = 'downloads'
            converted_video_path = os.path.join(output_folder, f"video_title.mp4")
            
            # Comando para convertir el video a formato .mp4

            ffmpeg_command = f'ffmpeg -i "{video_path}" "{converted_video_path}"'
            print(f"Ejecutando comando FFmpeg: {ffmpeg_command}")

            # Ejecuta el comando FFmpeg y captura la salida
            process = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)
            print("algo")
            
            if process.returncode != 0:
                # Si hubo un error en la conversión
                print(f"Error en FFmpeg: {process.stderr}")
            else:
                # Si la conversión fue exitosa
                print(f"Video convertido exitosamente: {converted_video_path}")

                model = load_model()
                output_filename = f"{video_title}_procesado.mp4"  # Usa el título del video como nombre de archivo
                detections, processed_video_path = process_video(converted_video_path, model, output_filename)

                if processed_video_path and os.path.exists(processed_video_path):
                    st.video(processed_video_path)

                    # Generar y mostrar informe y resumen
                    report_data = generate_report(detections)
                    report_df = pd.DataFrame(report_data)
                    st.table(report_df)

                    summary = generate_summary(detections)
                    st.markdown("### Summary")
                    st.markdown(summary)

                    if 'Class' in report_df.columns:
                        class_counts = report_df['Class'].value_counts()
                        st.bar_chart(class_counts)
        else:
            st.text("Error: Video not found.")



if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
