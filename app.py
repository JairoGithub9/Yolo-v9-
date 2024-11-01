import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Inicializar el modelo YOLO
model = YOLO('best.pt')

# Función para seleccionar y procesar la imagen
def select_image():
    global original_image_label, processed_image_label
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    if file_path:
        # Cargar y redimensionar la imagen original
        img = Image.open(file_path)
        img = img.resize((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        
        # Mostrar la imagen original
        original_image_label.config(image=img_tk)
        original_image_label.image = img_tk
        
        # Procesar la imagen con el modelo YOLO
        process_image(file_path)

def process_image(image_path):
    global processed_image_label
    
    # Redimensionar la imagen a 256x256
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    resized_image_path = "resized_image.jpg"
    cv2.imwrite(resized_image_path, img)
    
    # Procesar la imagen con el modelo YOLO
    results = model.predict(resized_image_path, imgsz=256, conf=0.3)
    
    # Dibujar las detecciones en la imagen
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.squeeze().cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = box.conf.item()

            # Obtener el nombre de la clase (asumiendo que la clase se llama 'fire')
            label = 'fire'

            # Dibujar el rectángulo del objeto detectado
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Añadir la etiqueta y la confianza
            cv2.putText(img, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Guardar la imagen procesada
    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, img)
    
    # Cargar y mostrar la imagen procesada
    processed_img = Image.open(processed_image_path)
    processed_img_tk = ImageTk.PhotoImage(processed_img)
    
    processed_image_label.config(image=processed_img_tk)
    processed_image_label.image = processed_img_tk

# Función para mostrar la pantalla principal
def show_main_screen():
    hide_all_frames()
    header_frame.pack(fill=tk.X, pady=10)
    menu_frame.pack(fill=tk.X, pady=10)
    content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    footer_frame.pack(fill=tk.X, pady=10)

# Función para mostrar la pantalla del manual
def show_manual_screen():
    hide_all_frames()
    header_frame.pack(fill=tk.X, pady=10)
    menu_frame.pack(fill=tk.X, pady=10)
    manual_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# Función para ocultar todos los frames
def hide_all_frames():
    header_frame.pack_forget()
    menu_frame.pack_forget()
    content_frame.pack_forget()
    footer_frame.pack_forget()
    manual_frame.pack_forget()

# Crear la ventana principal
root = tk.Tk()
root.title("Detección de Potenciales Focos de Incendio")
root.geometry("800x600")
root.configure(bg="#f4f4f4")

# Crear el encabezado
header_frame = tk.Frame(root, bg="#f4f4f4", bd=4, relief=tk.RAISED)
logo_img = Image.open("static/img/img1.png")
logo_img = logo_img.resize((100, 100))
logo_img_tk = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(header_frame, image=logo_img_tk, bg="#f4f4f4")
logo_label.pack(side=tk.LEFT, padx=10)
title_label = tk.Label(header_frame, text="Detección de Potenciales Focos de Incendio en Imágenes Aéreas",
                       font=("Arial", 18, "bold"), bg="#f4f4f4", fg="black")
title_label.pack(side=tk.LEFT, padx=10)

# Crear el menú
menu_frame = tk.Frame(root, bg="#f4f4f4", bd=4, relief=tk.RAISED)
btn_inicio = tk.Button(menu_frame, text="Inicio", font=("Arial", 12, "bold"), bg="#f4f4f4", fg="black", bd=0, command=show_main_screen)
btn_inicio.pack(side=tk.LEFT, padx=10)
btn_manual = tk.Button(menu_frame, text="Manual", font=("Arial", 12, "bold"), bg="#f4f4f4", fg="black", bd=0, command=show_manual_screen)
btn_manual.pack(side=tk.LEFT, padx=10)

# Crear el área de contenido principal
content_frame = tk.Frame(root, bg="#f4f4f4")
original_image_label = tk.Label(content_frame, bg="#f4f4f4")
original_image_label.pack(side=tk.LEFT, padx=20)
processed_image_label = tk.Label(content_frame, bg="#f4f4f4")
processed_image_label.pack(side=tk.LEFT, padx=20)

# Crear el pie de página
footer_frame = tk.Frame(root, bg="#f4f4f4", bd=4, relief=tk.RAISED)
btn_ingresar = tk.Button(footer_frame, text="INICIAR", font=("Arial", 14, "bold"), bg="#555", fg="white",
                         command=select_image, cursor="hand2")
btn_ingresar.pack(pady=10)

# Crear el frame del manual
manual_frame = tk.Frame(root, bg="#f4f4f4")
instructions_label = tk.Label(manual_frame, text="Manual de Instrucciones", font=("Arial", 16, "bold"), bg="#f4f4f4", fg="black")
instructions_label.pack(pady=10)

instructions_text = """
1. Abra la aplicación y haga clic en 'INICIAR' para cargar una imagen.
2. Seleccione una imagen desde su computadora.
3. La imagen se redimensionará automáticamente a 256x256 píxeles.
4. El modelo YOLO procesará la imagen para detectar posibles focos de incendio.
5. Las detecciones se mostrarán en la imagen con un cuadro delimitador y una etiqueta.
6. Puede guardar la imagen procesada con las detecciones.
"""
instructions_message = tk.Message(manual_frame, text=instructions_text, width=600, bg="#f4f4f4", font=("Arial", 12), fg="black")
instructions_message.pack(pady=10)

btn_back = tk.Button(manual_frame, text="Volver al Inicio", font=("Arial", 14, "bold"), bg="#555", fg="white",
                     command=show_main_screen, cursor="hand2")
btn_back.pack(pady=10)

# Mostrar la pantalla principal al inicio
show_main_screen()

# Ejecutar la aplicación
root.mainloop()


