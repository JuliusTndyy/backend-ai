import tensorflow as tf
import numpy as np
from PIL import Image

print("Memuat model .h5...")
model = tf.keras.models.load_model('eye_tiny_cnn_restored.h5')
CLASS_NAMES = ['Cataract', 'Conjunctivitis', 'Normal', 'Pterygium', 'Stye']

# Buka gambar sampel katarak lokal
try:
    img = Image.open('sampelKatark.jpg').convert('RGB')
    img = img.resize((96, 96))
    
    # Konversi ke array
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    print("Shape input:", img_array.shape)
    print("Nilai piksel min/max:", np.min(img_array), np.max(img_array))

    # Lakukan prediksi
    preds = model.predict(img_array)
    print("\nArray mentah hasil prediksi model:", preds)
    
    predicted_idx = np.argmax(preds[0])
    print(f"Hasil: {CLASS_NAMES[predicted_idx]} dengan skor mentah: {preds[0][predicted_idx]}")

except Exception as e:
    print("Error saat membaca sampel_katarak.jpg:", e)
    print("Tips: Pastikan file sampel_katarak.jpg sudah ditaruh di folder backend-ai!")