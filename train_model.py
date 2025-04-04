import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1. Load dataset
df = pd.read_csv('dataset_suhu/data_suhu.csv')  # Pastikan nama file sesuai

# 2. Menyesuaikan nama kolom agar sesuai dengan model yang diharapkan
df.rename(columns={
    'Suhu Rata-rata': 'Temperatur',
    'Kelembaban Relatif Rata rata': 'Kelembaban'
}, inplace=True)

# 3. Menangani data yang hilang
df.dropna(inplace=True)

# 4. Menampilkan beberapa data untuk memastikan perubahan sudah benar
print(df.head())

# 5. Pastikan semua kolom yang dibutuhkan ada
if 'Temperatur' not in df.columns or 'Kelembaban' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'Temperatur' dan 'Kelembaban'")

# 6. Menyiapkan fitur dan target
X = df [['Kelembaban']]
y = df['Temperatur']

# 7. Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Melatih model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Memprediksi data uji
y_pred = model.predict(X_test)

# 10. Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

# 11. Menyimpan model
joblib.dump(model, 'model.pkl')
print("Model berhasil disimpan sebagai 'model.pkl'")