import tensorflow as tf
import numpy as np
import cv2

print("Memuat model .h5...")
model = tf.keras.models.load_model('eye_tiny_cnn_restored.h5') # Sesuaikan nama file .h5 mu
model.summary() # Ini akan mencetak seluruh struktur layer ke terminal

CLASS_NAMES = ['Cataract', 'Conjunctivitis', 'Normal', 'Pterygium', 'Stye']

# Buat gambar dummy ukuran 96x96
dummy_img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

# Test dengan rentang 0-255 (Tanpa dibagi 255)
img_array_raw = np.expand_dims(dummy_img.astype(np.float32), axis=0)
pred_raw = model.predict(img_array_raw)

# Test dengan rentang 0.0 - 1.0 (Dibagi 255)
img_array_norm = np.expand_dims(dummy_img.astype(np.float32) / 255.0, axis=0)
pred_norm = model.predict(img_array_norm)

print("\n--- HASIL TEST DUMMY ---")
print("Prediksi Mentah (0-255) max score:", np.max(pred_raw))
print("Prediksi Normalisasi (0-1) max score:", np.max(pred_norm))