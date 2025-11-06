import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Muat dataset
url = 'https://raw.githubusercontent.com/MainakRepositor/Datasets/master/lithium-ion%20batteries.csv'
data = pd.read_csv(url)


# Pilih kolom numerik yang relevan
features = ['Formation Energy (eV)', 'E Above Hull (eV)', 'Band Gap (eV)', 'Density (gm/cc)', 'Volume']
data_numeric = data[features]

# Normalisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric.fillna(data_numeric.mean()))

# Pisahkan data menjadi set pelatihan dan pengujian
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# PCA
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Hitung jarak Mahalanobis sebagai skor anomali
mean_vec = np.mean(X_train_pca, axis=0)
cov_mat = np.cov(X_train_pca, rowvar=False)
inv_cov_mat = np.linalg.inv(cov_mat)
distances = [np.sqrt((x - mean_vec).T.dot(inv_cov_mat).dot(x - mean_vec)) for x in X_test_pca]

# Tentukan ambang batas berdasarkan persentil
threshold = np.percentile(distances, 95)

# Deteksi anomali
anomalies_pca = np.array(distances) > threshold

# Plot hasil PCA
fig_pca, ax_pca = plt.subplots()
scatter_pca = ax_pca.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=anomalies_pca, cmap='coolwarm')
ax_pca.set_title("Deteksi Anomali menggunakan PCA")
ax_pca.set_xlabel("Komponen Utama 1")
ax_pca.set_ylabel("Komponen Utama 2")
ax_pca.legend(*scatter_pca.legend_elements(), title="Anomali")

# Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
isolation_forest.fit(X_train)
anomalies_iforest = isolation_forest.predict(X_test) == -1

# Plot hasil Isolation Forest
fig_iforest, ax_iforest = plt.subplots()
scatter_iforest = ax_iforest.scatter(X_test[:, 0], X_test[:, 1], c=anomalies_iforest, cmap='coolwarm')
ax_iforest.set_title("Deteksi Anomali menggunakan Isolation Forest")
ax_iforest.set_xlabel("Formation Energy (eV)")
ax_iforest.set_ylabel("E Above Hull (eV)")
ax_iforest.legend(*scatter_iforest.legend_elements(), title="Anomali")

# Tampilan Streamlit
st.title("Deteksi Anomali pada Dataset Baterai Lithium-ion")
st.write("""
Aplikasi ini mendemonstrasikan deteksi anomali pada dataset baterai lithium-ion menggunakan 
Principal Component Analysis (PCA) dan Isolation Forest. Tujuannya adalah untuk mengidentifikasi data 
yang menyimpang secara signifikan dari mayoritas data, yang menunjukkan potensi anomali.
""")

# Tampilkan kolom dataset di Streamlit
st.write("Kolom-kolom dalam dataset:")
st.write(data.columns)

# Penjelasan tentang PCA
st.header("Deteksi Anomali menggunakan PCA")
st.write("""
Principal Component Analysis (PCA) adalah teknik reduksi dimensi yang mengubah 
data ke dalam sistem koordinat baru di mana variansi terbesar oleh proyeksi 
data terletak pada komponen utama pertama, variansi terbesar kedua pada 
komponen utama kedua, dan seterusnya. 

Dalam aplikasi ini, kami mengurangi dimensi dataset menjadi 2 komponen utama. 
Kami kemudian menghitung jarak Mahalanobis dari setiap titik data uji dari 
mean titik data pelatihan di ruang PCA baru. Titik data dengan jarak lebih besar 
dari ambang batas tertentu (persentil ke-95) dianggap sebagai anomali.
""")
st.pyplot(fig_pca)
st.write(f"Jumlah anomali terdeteksi menggunakan PCA: {np.sum(anomalies_pca)}")
st.write("Indeks anomali menggunakan PCA:", np.where(anomalies_pca)[0])

# Penjelasan tentang Isolation Forest
st.header("Deteksi Anomali menggunakan Isolation Forest")
st.write("""
Isolation Forest adalah teknik pembelajaran ensemble yang dirancang khusus untuk deteksi anomali. 
Ia mengisolasi observasi dengan memilih fitur secara acak dan kemudian memilih nilai split secara 
acak antara nilai maksimum dan minimum dari fitur yang dipilih. 

Proses isolasi ini dapat direpresentasikan oleh struktur pohon, dan jumlah split yang diperlukan 
untuk mengisolasi suatu titik data setara dengan panjang jalur dari node akar ke node terminasi. 
Anomali adalah observasi dengan jalur terpendek, karena mereka sedikit dan berbeda.

Dalam aplikasi ini, kami melatih Isolation Forest pada data pelatihan dan kemudian memprediksi 
anomali pada data uji. Titik data yang diprediksi sebagai anomali ditandai dengan warna yang berbeda.
""")
st.pyplot(fig_iforest)
st.write(f"Jumlah anomali terdeteksi menggunakan Isolation Forest: {np.sum(anomalies_iforest)}")
st.write("Indeks anomali menggunakan Isolation Forest:", np.where(anomalies_iforest)[0])

st.write("""
Dataset yang digunakan dalam aplikasi ini dapat ditemukan 
[di sini](https://raw.githubusercontent.com/MainakRepositor/Datasets/master/lithium-ion%20batteries.csv).
""")
