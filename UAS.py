# 1. Import library yang diperlukan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Import dataset
data = pd.read_csv(r'D:\Machine Learning\DataMalware.csv')  # Pastikan file path sesuai dengan lokasi file Anda

# 3. Preprocessing Data
# 3.1 Mengatasi nilai yang hilang
data['MajorOperatingSystemVersion'] = data['MajorOperatingSystemVersion'].fillna(data['MajorOperatingSystemVersion'].mean())
data['ResourceSize'] = data['ResourceSize'].fillna(data['ResourceSize'].mean())

# 3.2 Mengencode kolom target 'legitimate'
data['legitimate'] = data['legitimate'].map({'malware': 1, 'not malware': 0})

# 3.3 Memisahkan fitur dan target
X = data.drop(columns=['legitimate'])  # Fitur
y = data['legitimate']  # Target

# 3.4 Normalisasi data numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Melakukan normalisasi pada fitur

# 4. Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)  # Pisahkan data dengan perbandingan 70% latih dan 30% uji

# 5. Implementasi Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)  # Membuat model Random Forest
rf_model.fit(X_train, y_train)  # Melatih model dengan data latih

# 6. Prediksi dengan model Random Forest
y_pred_rf = rf_model.predict(X_test)  # Prediksi hasil pada data uji

# 7. Evaluasi Kinerja Model
print("Evaluasi Random Forest:")
print(f"Akurasi: {accuracy_score(y_test, y_pred_rf):.4f}")  # Menampilkan akurasi model dengan format 4 desimal
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred_rf))  # Menampilkan laporan klasifikasi
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))  # Menampilkan confusion matrix

# 8. Visualisasi

# 8.1 Visualisasi distribusi nilai fitur dengan pengaturan font kecil
features = data.columns  # Ambil nama semua fitur
num_features = len(features)  # Hitung jumlah fitur
rows = (num_features + 2) // 3  # Tentukan jumlah baris (maks 3 kolom per baris)

fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(18, 4 * rows))  # Membuat grid dinamis sesuai jumlah fitur
axes = axes.flatten()  # Ubah menjadi array untuk iterasi yang lebih mudah

# Plot histogram untuk setiap fitur
for i, feature in enumerate(features):
    ax = axes[i]
    ax.hist(data[feature], bins=30, color='skyblue', edgecolor='black')  # Histogram dengan warna
    ax.set_title(feature, fontsize=10, pad=5)  # Judul tiap subplot dengan ukuran font kecil
    ax.set_xlabel('Nilai', fontsize=8, labelpad=5)  # Label sumbu x dengan ukuran font kecil
    ax.set_ylabel('Frekuensi', fontsize=8, labelpad=5)  # Label sumbu y dengan ukuran font kecil
    ax.tick_params(axis='x', labelsize=6, rotation=45)  # Rotasi label x dengan ukuran font lebih kecil
    ax.tick_params(axis='y', labelsize=6)  # Ukuran label y lebih kecil
    ax.grid(True, linestyle='--', alpha=0.7)  # Tambahkan grid

# Hapus subplot kosong jika jumlah subplot lebih banyak dari jumlah fitur
for j in range(num_features, len(axes)):
    fig.delaxes(axes[j])

# Tambahkan judul utama
plt.suptitle('Distribusi Nilai Fitur', fontsize=12, y=0.95)  # Ukuran font lebih kecil untuk judul utama
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.6, wspace=0.4)  # Tambah spasi antar subplots
plt.show()

# 8.2 Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))

# Menampilkan confusion matrix dengan heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Malware', 'Malware'], yticklabels=['Not Malware', 'Malware'])

# Mengatur judul dan label dengan font yang lebih kecil
plt.title('Confusion Matrix', fontsize=10, pad=8)
plt.xlabel('Prediksi', fontsize=8, labelpad=5)
plt.ylabel('Aktual', fontsize=8, labelpad=5)

# Mengatur font untuk angka dalam heatmap
plt.tick_params(axis='both', which='major', labelsize=6)
plt.tight_layout()  # Mengatur agar tidak ada elemen yang terpotong
plt.show()
