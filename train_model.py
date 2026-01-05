import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset_kelulusan.csv")

# Encoder
encoder_status = LabelEncoder()
encoder_pekerjaan = LabelEncoder()
encoder_kehadiran = LabelEncoder()

df["status kelulusan"] = encoder_status.fit_transform(df["status kelulusan"])
df["pekerjaan sambil kuliah"] = encoder_pekerjaan.fit_transform(df["pekerjaan sambil kuliah"])
df["kategori kehadiran"] = encoder_kehadiran.fit_transform(df["kategori kehadiran"])

# Fitur & target
X = df.drop("status kelulusan", axis=1)
y = df["status kelulusan"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔑 Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Simpan model & encoder
joblib.dump(model, "model_kelulusan.pkl")
joblib.dump(encoder_status, "label_encoder.pkl")
joblib.dump(encoder_pekerjaan, "encoder_pekerjaan.pkl")
joblib.dump(encoder_kehadiran, "encoder_kehadiran.pkl")

print("✅ Model Decision Tree berhasil disimpan")
