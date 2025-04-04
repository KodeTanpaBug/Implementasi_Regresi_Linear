from flask import Flask, render_template, request
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model regresi linear (dummy, nanti dilatih dari CSV)
model = LinearRegression()
model_trained = False  # Status apakah model sudah dilatih

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_trained
    prediction = None
    data_table = None
    pred_table = None
    
    if request.method == 'POST':
        # Input manual
        try:
            suhu_min = float(request.form['suhu_min'])
            suhu_max = float(request.form['suhu_max'])
            suhu_avg = float(request.form['suhu_avg'])
            kelembaban = float(request.form['kelembaban'])
            curah_hujan = float(request.form['curah_hujan'])
            durasi_sinar = float(request.form['durasi_sinar'])
            
            # Pastikan model sudah dilatih sebelum prediksi
            if model_trained:
                input_data = np.array([[suhu_min, suhu_max, suhu_avg, kelembaban, curah_hujan, durasi_sinar]])
                prediction = model.predict(input_data)[0]
            else:
                prediction = "Error: Model belum dilatih, unggah dataset terlebih dahulu."
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction, data_table=data_table, pred_table=pred_table)

@app.route('/upload', methods=['POST'])
def upload_file():
    global model_trained
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Baca CSV
        df = pd.read_csv(filepath)
        
        # Mapping nama kolom agar sesuai dengan dataset pengguna
        column_mapping = {
            'Suhu Minimum': 'Suhu Min',
            'Suhu Maksimum': 'Suhu Max',
            'Suhu Rata-rata': 'Suhu Avg',
            'Kelembaban Relatif Rata rata': 'Kelembaban',
            'Curah Hujan': 'Curah Hujan',
            'Durasi Penyinaran Matahari (dalam jam)': 'Durasi Sinar'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        required_columns = list(column_mapping.values())
        
        # Pastikan semua kolom yang dibutuhkan ada dalam dataset
        if not all(col in df.columns for col in required_columns):
            return f"CSV harus memiliki kolom: {', '.join(column_mapping.keys())}"
        
        # Latih model regresi linear
        X = df[required_columns]
        y = df['Suhu Avg']  # Misalkan kita ingin memprediksi suhu rata-rata
        model.fit(X, y)
        model_trained = True  # Tandai bahwa model sudah dilatih
        
        # Lakukan prediksi
        df['Prediksi Suhu'] = model.predict(X)
        
        # Konversi ke HTML untuk ditampilkan di tabel
        data_table = df[required_columns].head().to_html(classes='table table-striped', index=False)
        pred_table = df[['Prediksi Suhu']].head().to_html(classes='table table-striped', index=False)
        
        return render_template('index.html', data_table=data_table, pred_table=pred_table)

if __name__ == '__main__':
    app.run(debug=True)
