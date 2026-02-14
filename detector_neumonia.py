#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import csv
import tkcap
import os

from servicio.read_img import read_dicom_file, read_jpg_file
from servicio.integrator import obtener_prediccion

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Detección de Neumonía - UAO")
        fonti = font.Font(weight="bold")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # LABELS
        ttk.Label(self.root, text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", font=fonti).place(x=122, y=25)
        ttk.Label(self.root, text="Imagen Radiográfica", font=fonti).place(x=110, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=fonti).place(x=545, y=65)
        ttk.Label(self.root, text="Cédula Paciente:", font=fonti).place(x=65, y=350)
        ttk.Label(self.root, text="Resultado:", font=fonti).place(x=500, y=350)
        ttk.Label(self.root, text="Probabilidad:", font=fonti).place(x=500, y=400)

        # WIDGETS
        self.ID = StringVar()
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=15)
        self.text1.place(x=200, y=350)
        
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img1.place(x=65, y=90)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text_img2.place(x=500, y=90)
        
        self.text2 = Text(self.root)
        self.text2.place(x=610, y=350, width=110, height=30)
        self.text3 = Text(self.root)
        self.text3.place(x=610, y=400, width=90, height=30)

        # BOTONES
        self.button2 = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button2.place(x=70, y=460)
        self.button1 = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button1.place(x=220, y=460)
        self.button6 = ttk.Button(self.root, text="Guardar CSV", command=self.save_results_csv)
        self.button6.place(x=370, y=460)
        self.button4 = ttk.Button(self.root, text="Guardar PDF", command=self.create_pdf)
        self.button4.place(x=520, y=460)
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button3.place(x=670, y=460)

        self.array = None
        self.label = ""
        self.proba = 0.0
        self.root.mainloop()

    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("Todos los archivos", "*.*"), ("DICOM", "*.dcm"), ("JPG", "*.jpg"), ("PNG", "*.png"))
        )
        if filepath:
            if filepath.lower().endswith('.dcm'):
                self.array, img2show = read_dicom_file(filepath)
            else:
                self.array, img2show = read_jpg_file(filepath)
                
            self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.delete('1.0', END)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "normal"

    def run_model(self):
        self.label, self.proba, heatmap_array = obtener_prediccion(self.array)
        
        self.img2 = Image.fromarray(heatmap_array)
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        
        self.text_img2.delete('1.0', END)
        self.text_img2.image_create(END, image=self.img2)
        
        self.text2.delete('1.0', END)
        self.text2.insert(END, self.label)
        
        self.text3.delete('1.0', END)
        self.text3.insert(END, f"{self.proba:.2f}%")

    def save_results_csv(self):
        if not self.label:
            showinfo(title="Aviso", message="Debes predecir antes de guardar.")
            return
        
        try:
            # Usamos 'a' para append (añadir)
            with open("historial.csv", "a", newline='', encoding='utf-8') as csvfile:
                w = csv.writer(csvfile, delimiter=";") # Cambiado a ; para mejor compatibilidad con Excel
                w.writerow([self.text1.get(), self.label, f"{self.proba:.2f}%"])
                csvfile.flush() # <--- ESTO fuerza a escribir en el disco
            
            showinfo(title="Guardar", message=f"Datos guardados en historial.csv\nPaciente: {self.text1.get()}")
        except Exception as e:
            showinfo(title="Error", message=f"No se pudo guardar: {e}")

    def create_pdf(self):
        cedula = self.text1.get().strip() or "SinCedula"
        pdf_path = f"Reporte_{cedula}.pdf"
        img_temp = "temp_capture.jpg"
        
        # Manejo de nombres duplicados
        count = 1
        while os.path.exists(pdf_path):
            pdf_path = f"Reporte_{cedula}_{count}.pdf"
            count += 1

        cap = tkcap.CAP(self.root)
        cap.capture(img_temp)
        img = Image.open(img_temp).convert("RGB")
        img.save(pdf_path)
        os.remove(img_temp)
        showinfo(title="PDF", message=f"PDF generado: {pdf_path}")

    def delete(self):
        if askokcancel(title="Borrar", message="¿Desea limpiar todo?"):
            self.text1.delete(0, END)
            self.text2.delete('1.0', END)
            self.text3.delete('1.0', END)
            self.text_img1.delete('1.0', END)
            self.text_img2.delete('1.0', END)
            self.button1["state"] = "disabled"
            self.label = ""

if __name__ == "__main__":
    App()