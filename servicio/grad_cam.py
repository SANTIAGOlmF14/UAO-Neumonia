import cv2
import numpy as np
import tensorflow as tf
from servicio.preprocess_img import preprocess

def aplicar_grad_cam(array, model):
    # 1. Preprocesar la imagen
    # Asegúrate que preprocess devuelva un float32
    img_tensor = preprocess(array) 
    
    # 2. Configurar la capa objetivo (la última convolucional antes del pooling)
    layer_name = "conv10_thisone"
    
    # Construir el modelo de gradientes
    # Usamos model.input en lugar de [model.inputs] para evitar warnings de anidación
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, # Con 's' al final
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # 3. Grabar las operaciones para calcular gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        # CORRECCIÓN: Si predictions es una lista, extraemos el tensor
        if isinstance(predictions, list):
            predictions = predictions[0]
            
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # 4. Calcular gradientes
    # CORRECCIÓN: Extraemos el tensor de conv_outputs si es lista
    grads = tape.gradient(loss, conv_outputs)
    if isinstance(grads, list):
        grads = grads[0]
    
    if isinstance(conv_outputs, list):
        conv_outputs = conv_outputs[0]

    # 5. Calcular el peso de cada canal (Global Average Pooling de los gradientes)
    # grads tiene forma (batch, height, width, channels), promediamos sobre H y W
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Generar el mapa de calor
    # Multiplicamos la salida de la capa conv por los pesos de importancia
    output = conv_outputs[0] # Quitamos la dimensión del batch -> (H, W, C)
    heatmap = output @ weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # Normalización ReLU y escala 0-1
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # 7. Procesamiento visual
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Preparar imagen original
    orig_img = cv2.resize(array, (512, 512))
    
    # Asegurar que la imagen original sea uint8 y tenga 3 canales (RGB)
    if orig_img.dtype != np.uint8:
        # Si la imagen viene normalizada (0-1), pasar a 0-255
        if np.max(orig_img) <= 1.0:
            orig_img = (orig_img * 255).astype(np.uint8)
        else:
            orig_img = orig_img.astype(np.uint8)

    if len(orig_img.shape) == 2: # Si es Grayscale
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    elif orig_img.shape[2] == 1: # Si tiene un canal pero forma (H,W,1)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    
    # Mezclar imagen con el mapa de calor
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
    
    # Devolver las predicciones como numpy array para la interfaz
    return superimposed_img, predictions.numpy()