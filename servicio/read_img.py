import pydicom
from PIL import Image
import numpy as np

def read_dicom_file(path):
    ds = pydicom.dcmread(path)
    img_array = ds.pixel_array.astype(float)
    # Normalización simple para visualización
    rescaled_image = (np.maximum(img_array, 0) / img_array.max()) * 255
    rescaled_image = np.uint8(rescaled_image)
    final_image = Image.fromarray(rescaled_image)
    return img_array, final_image

def read_jpg_file(path):
    img = Image.open(path).convert('L') # Convertir a escala de grises
    img_array = np.array(img)
    return img_array, img