import numpy as np
from PIL import Image
import cv2

def preprocess(array):
    # 1. Asegurar escala de grises
    if len(array.shape) == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    
    # 2. Redimensionar usando OpenCV con interpolación de área (mejor para CNNs profundas)
    # El tamaño 512, 512 es el que pide tu capa 'input_9'
    img_resized = cv2.resize(array, (512, 512), interpolation=cv2.INTER_AREA)
    
    # 3. Normalización estricta [0, 1]
    # Si la imagen ya es float, la llevamos a 0-1. Si es uint8, dividimos por 255.
    img_array = img_resized.astype('float32')
    if img_array.max() > 1.0:
        img_array /= 255.0
    
    # 4. Ajustar dimensiones (1, 512, 512, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array