import numpy as np
from servicio.load_model import cargar_modelo
from servicio.grad_cam import aplicar_grad_cam

MODELO_SISTEMA = cargar_modelo()

def obtener_prediccion(array):
    heatmap_img, preds = aplicar_grad_cam(array, MODELO_SISTEMA)
    
    if isinstance(preds, list):
        preds = preds[0]
    
    # Ver las probabilidades en consola ayuda a saber si el modelo duda
    print(f"Probabilidades brutas (0:Bact, 1:Norm, 2:Viral): {preds[0]}")
    
    idx = np.argmax(preds[0])
    probabilidad = preds[0][idx] * 100
    
    # ESTE ES EL ORDEN ALFABÉTICO QUE KERAS ASIGNA POR DEFECTO:
    # B - Bacterial (0)
    # N - Normal (1)
    # V - Viral (2)
    clases = ["Bacteriana", "Normal", "Viral"] 
    
    etiqueta = clases[idx]
    heatmap_img_rgb = heatmap_img[:, :, ::-1]
    
    return etiqueta, probabilidad, heatmap_img_rgb