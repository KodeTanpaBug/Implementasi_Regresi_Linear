from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd  # Tambahkan ini untuk membaca CSV
import os

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Folder untuk menyimpan file yang diunggah sementara
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None

    suhu_min_value = ""
    suhu_max_value = ""
    suhu_avg_value = ""
    kelembaban_value = ""
    curah_hujan_value = ""
    durasi_sinar_value = ""

    if request.method == "POST":
        try:
            suhu_min_value = request.form.get("suhu_min", "")
            suhu_max_value = request.form.get("suhu_max", "")
            suhu_avg_value = request.form.get("suhu_avg", "")
            kelembaban_value = request.form.get("kelembaban", "")
            curah_hujan_value = request.form.get("curah_hujan", "")
            durasi_sinar_value = request.form.get("durasi_sinar", "")

            if not all([suhu_min_value, suhu_max_value, suhu_avg_value, kelembaban_value, curah_hujan_value, durasi_sinar_value]):
                raise ValueError("Semua field harus diisi!")

            kelembaban_value = float(kelembaban_value)
            features = np.array([[kelembaban_value]])
            prediction = model.predict(features)[0]
            prediction = round(prediction, 2)

        except ValueError as e:
            error_message = f"Terjadi kesalahan: {str(e)}"

    return render_template("index.html",
                           prediction=prediction,
                           error_message=error_message,
                           suhu_min_value=suhu_min_value,
                           suhu_max_value=suhu_max_value,
                           suhu_avg_value=suhu_avg_value,
                           kelembaban_value=kelembaban_value,
                           curah_hujan_value=curah_hujan_value,
                           durasi_sinar_value=durasi_sinar_value)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "Tidak ada file yang diunggah", 400

    file = request.files["file"]

    if file.filename == "":
        return "Nama file tidak valid", 400

    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Baca file CSV
        df = pd.read_csv(file_path)

        # Konversi DataFrame ke format HTML
        table_html = df.to_html(classes="table table-bordered table-striped", index=False)

        return render_template("index.html", table_html=table_html)

if __name__ == "__main__":
    app.run(debug=True)