import os
import tensorflow as tf

def cargar_modelo():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'model', 'conv_MLP_84.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
        
    try:
        # CORRECCIÓN: compile=False evita que Keras busque el error de 'reduction=auto'
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Modelo cargado exitosamente (modo inferencia) desde: {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"Error técnico al cargar el modelo: {e}")