from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD MODEL & ENCODER
# =========================
model = joblib.load("model_kelulusan.pkl")
encoder_pekerjaan = joblib.load("encoder_pekerjaan.pkl")
encoder_kehadiran = joblib.load("encoder_kehadiran.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # =========================
    # AMBIL & VALIDASI INPUT
    # =========================
    ipk = max(0, min(float(request.form.get("ipk", 0)), 4.0))
    mk_tidak_lulus = int(request.form.get("mk_tidak_lulus", 0))
    cuti = int(request.form.get("cuti", 0))
    semester = int(request.form.get("semester", 1))

    ips_rata = float(request.form.get("ips_rata", 0))
    ips_akhir = float(request.form.get("ips_akhir", 0))
    ips_tren = float(request.form.get("ips_tren", 0))

    pekerjaan = request.form.get("pekerjaan", "Tidak")
    kehadiran = request.form.get("kehadiran", "Sedang")

    # =========================
    # ENCODING
    # =========================
    pekerjaan_enc = (
        encoder_pekerjaan.transform([pekerjaan])[0]
        if pekerjaan in encoder_pekerjaan.classes_
        else encoder_pekerjaan.transform(["Tidak"])[0]
    )

    kehadiran_enc = (
        encoder_kehadiran.transform([kehadiran])[0]
        if kehadiran in encoder_kehadiran.classes_
        else encoder_kehadiran.transform(["Sedang"])[0]
    )

    # =========================
    # DATAFRAME
    # =========================
    df = pd.DataFrame([{
        "ipk": ipk,
        "mata kuliah tidak lulus": mk_tidak_lulus,
        "jumlah cuti akademik": cuti,
        "pekerjaan sambil kuliah": pekerjaan_enc,
        "jumlah semester": semester,
        "ips rata-rata": ips_rata,
        "ips semester akhir": ips_akhir,
        "ips tren": ips_tren,
        "kategori kehadiran": kehadiran_enc
    }])

    # =========================
    # PROBABILITAS
    # =========================
    proba = model.predict_proba(df)[0]

    prob_tidak = round(proba[0] * 100, 2)  # class 0
    prob_lulus = round(proba[1] * 100, 2)  # class 1

    # =========================
    # KEPUTUSAN AKADEMIK
    # =========================
    THRESHOLD_LULUS = 60  # %

    hasil = "Lulus Tepat Waktu" if prob_lulus >= THRESHOLD_LULUS else "Tidak Lulus Tepat Waktu"

    # =========================
    # KIRIM KE HTML
    # =========================
    return render_template(
        "index.html",
        hasil=hasil,
        prob_lulus=prob_lulus,
        prob_tidak=prob_tidak,

        ipk=ipk,
        mk_tidak_lulus=mk_tidak_lulus,
        cuti=cuti,
        pekerjaan=pekerjaan,
        semester=semester,
        ips_rata=ips_rata,
        ips_akhir=ips_akhir,
        ips_tren=ips_tren,
        kehadiran=kehadiran
    )


if __name__ == "__main__":
    app.run(debug=True)
