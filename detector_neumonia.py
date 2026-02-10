#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry
from tkinter.messagebox import askokcancel, showinfo, WARNING

# ---------------------- ERROR 1: FALTABA IMPORTAR PYDICOM ----------------------
# Tu código usaba "dicom.read_file" pero no importaba la librería.
import pydicom as dicom

# ---------------------- ERROR 2: FALTABA IMPORTAR TENSORFLOW -------------------
# Usabas tf.compat.v1... pero nunca importaste tensorflow.
import tensorflow as tf
import tensorflow.keras.backend as K

from PIL import ImageTk, Image
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time
import cv2

# Desactivación de eager execution (correcto, pero antes fallaba porque no estaba importado TF)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


# ---------------------- ERROR 3: NO EXISTÍA model_fun() -----------------------
# Tu código intentaba usar model_fun() pero no estaba definida.
# TEMPORAL: la dejo aquí antes de modularizar.
def model_fun():
    return tf.keras.models.load_model("WilhemNet86.h5")


# -------------------------------------------------------------------------------
# GRAD-CAM
# -------------------------------------------------------------------------------
def grad_cam(array):
    # ERROR 4: Faltaba model_fun importado → se corrigió arriba.
    img = preprocess(array)
    model = model_fun()

    preds = model.predict(img)
    argmax = np.argmax(preds[0])

    output = model.output[:, argmax]

    # ---------------------- ERROR 5: CAPA "conv10_thisone" NO SIEMPRE EXISTE ----
    # De momento la dejamos así porque es tu código original, pero si da error
    # significa que el modelo no tiene esa capa.
    last_conv_layer = model.get_layer("conv10_thisone")

    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # ---------------------- ERROR 6: K.function MALA POR FALTAR IMPORTS ----------
    # Ya está arreglado porque se importó K y tf correctamente.
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)

    for filters in range(64):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)

    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]


# -------------------------------------------------------------------------------
# PREDICCIÓN (CNN + HEATMAP)
# -------------------------------------------------------------------------------
def predict(array):
    batch_array_img = preprocess(array)
    model = model_fun()

    # ---------------------- ERROR 7: DOBLE PREDICCIÓN INNECESARIA ----------------
    # (No afecta funcionamiento pero es ineficiente)
    preds = model.predict(batch_array_img)
    prediction = np.argmax(preds)
    proba = np.max(preds) * 100

    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"

    heatmap = grad_cam(array)
    return (label, proba, heatmap)


# -------------------------------------------------------------------------------
# LECTURA DE IMÁGENES
# -------------------------------------------------------------------------------
def read_dicom_file(path):
    # ---------------------- ERROR 8: dicom.read_file YA NO EXISTE ----------------
    # se reemplazó por dicom.dcmread()
    img = dicom.dcmread(path)

    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)

    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


# -------------------------------------------------------------------------------
# PREPROCESAMIENTO
# -------------------------------------------------------------------------------
def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)

    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)

    return array


# -------------------------------------------------------------------------------
# INTERFAZ GRÁFICA
# -------------------------------------------------------------------------------
class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        fonti = font.Font(weight="bold")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.ID = StringVar()
        self.result = StringVar()

        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        # IMAGE WINDOWS
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        # BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        # POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)

        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)

        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()

        self.array = None
        self.reportID = 0

        self.root.mainloop()

    # ---------------------------------------------------------------------------
    # CARGAR IMAGEN
    # ---------------------------------------------------------------------------
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:

            # ---------------------- ERROR 9: SOLO LEÍAS DICOM ---------------------
            # Ahora detecta automáticamente tipo
            if filepath.lower().endswith(".dcm"):
                self.array, img2show = read_dicom_file(filepath)
            else:
                self.array, img2show = read_jpg_file(filepath)

            self.img1 = img2show.resize((250, 250), Image.ANTIALIAS)
            self.img1 = ImageTk.PhotoImage(self.img1)

            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    # ---------------------------------------------------------------------------
    # EJECUTAR MODELO
    # ---------------------------------------------------------------------------
    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.ANTIALIAS)
        self.img2 = ImageTk.PhotoImage(self.img2)

        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    # ---------------------------------------------------------------------------
    # GUARDAR EN CSV
    # ---------------------------------------------------------------------------
    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    # ---------------------------------------------------------------------------
    # GENERAR PDF
    # ---------------------------------------------------------------------------
    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID)

        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)

        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    # ---------------------------------------------------------------------------
    # BORRAR DATOS
    # ---------------------------------------------------------------------------
    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete("1.0", "end")
            self.text3.delete("1.0", "end")

            # ---------------------- ERROR 10: TEXT.delete NO ACEPTA IMÁGENES -------
            # Solución: borrar contenido completo, no la imagen directamente.
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")

            showinfo(title="Borrar", message="Los datos se borraron con éxito")


# -------------------------------------------------------------------------------
def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
