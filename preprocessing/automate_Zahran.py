import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def run_preprocessing():
    print("Memulai proses otomatisasi data...")

    # 1. Load Data Mentah
    # Path ini disesuaikan karena script akan dieksekusi dari luar folder preprocessing
    raw_data_path = 'dataset_raw/data_mentah.csv'
    
    if not os.path.exists(raw_data_path):
        print(f"Error: File tidak ditemukan di path: {raw_data_path}")
        return

    df = pd.read_csv(raw_data_path)
    print("Data berhasil dimuat. Mulai membersihkan missing values...")
    
    # 2. Mengatasi Missing Values
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # 3. Label Encoding
    print("Melakukan encoding pada data teks...")
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # 4. Simpan Data Bersih
    output_dir = 'preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/dataset_clean.csv'
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai! Data bersih berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()