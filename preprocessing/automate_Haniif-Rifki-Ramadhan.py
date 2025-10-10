import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    """
    Memuat dataset dari file CSV
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data berhasil dimuat dari {file_path}")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File {file_path} tidak ditemukan!")
        return None
    except Exception as e:
        print(f"✗ Error saat memuat data: {str(e)}")
        return None


def drop_unnecessary_columns(df, columns_to_drop=['Vol.', 'Change %']):
    """
    Menghapus kolom yang tidak diperlukan
    """
    df = df.drop(columns_to_drop, axis=1)
    print(f"✓ Kolom {columns_to_drop} berhasil dihapus")
    return df


def convert_and_sort_date(df, date_column='Date'):
    """
    Mengonversi kolom tanggal ke tipe datetime dan mengurutkan data
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    print(f"✓ Kolom '{date_column}' dikonversi ke datetime dan data diurutkan")
    return df


def handle_missing_values(df):
    """
    Menangani nilai yang hilang dengan menghapusnya
    """
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_after = df.isnull().sum().sum()
    print(f"✓ Missing values dihapus: {missing_before} → {missing_after}")
    return df


def rename_columns(df, rename_dict={'Price': 'Close'}):
    """
    Mengubah nama kolom
    """
    df = df.rename(columns=rename_dict)
    print(f"✓ Kolom berhasil direname: {rename_dict}")
    return df


def set_date_as_index(df, date_column='Date'):
    """
    Mengatur kolom tanggal sebagai index
    """
    df = df.set_index(date_column)
    print(f"✓ Kolom '{date_column}' dijadikan index")
    return df


def clean_price_columns(df, price_columns=['Close', 'Open', 'High', 'Low']):
    """
    Membersihkan kolom harga dari koma dan mengonversi ke float
    """
    for col in price_columns:
        if col in df.columns:
            # Hapus koma
            df[col] = df[col].str.replace(',', '')
            # Konversi ke float
            df[col] = df[col].astype(float)
    print(f"✓ Kolom harga dibersihkan dan dikonversi ke float: {price_columns}")
    return df


def scale_data(df, feature_column='Close', test_year=2022):
    """
    Melakukan scaling pada data menggunakan MinMaxScaler
    dan membagi menjadi train dan test
    """
    # Mengambil data untuk scaling
    data = df[[feature_column]].copy()
    
    # Melakukan scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Membuat DataFrame baru dengan data yang sudah di-scale
    df_scaled = df.copy()
    df_scaled[f'{feature_column}_scaled'] = scaled_data
    
    print(f"✓ Data berhasil di-scale dengan MinMaxScaler")
    print(f"  Range: [0, 1]")
    
    return df_scaled, scaler


def create_sequences(df, feature_column='Close_scaled', window_size=60, test_year=2022):
    """
    Membuat sequences untuk LSTM
    Memisahkan train dan test berdasarkan tahun
    """
    # Memastikan data terurut berdasarkan index tanggal
    df = df.sort_index()
    
    # Ambil data scaled
    data = df[feature_column].values
    
    # Membagi data berdasarkan tahun test
    test_size = df[df.index.year == test_year].shape[0]
    train_data = data[:-test_size]
    test_data = data[-test_size - window_size:]
    
    print(f"✓ Data dibagi menjadi train dan test")
    print(f"  Train size: {len(train_data)}")
    print(f"  Test size: {test_size}")
    
    # Fungsi helper untuk membuat sequences
    def make_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    # Membuat sequences untuk train
    X_train, y_train = make_sequences(train_data, window_size)
    
    # Membuat sequences untuk test
    X_test, y_test = make_sequences(test_data, window_size)
    
    print(f"✓ Sequences berhasil dibuat")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def save_preprocessed_data(df, X_train, y_train, X_test, y_test, scaler, output_dir):
    """
    Menyimpan data yang sudah diproses dalam format NPZ dan CSV
    """
    try:
        # Buat folder jika belum ada
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Folder '{output_dir}' dibuat")
        
        # Simpan DataFrame yang sudah di-scale
        csv_path = os.path.join(output_dir, 'Gold Price (2013-2023)_preprocessing.csv')
        df.to_csv(csv_path)
        print(f"✓ DataFrame berhasil disimpan ke {csv_path}")
        
        # Simpan sequences dalam format NPZ
        npz_path = os.path.join(output_dir, 'Gold Price (2013-2023)_sequences.npz')
        np.savez(
            npz_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        print(f"✓ Sequences berhasil disimpan ke {npz_path}")
        
        # Simpan scaler untuk digunakan saat inference
        import pickle
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Scaler berhasil disimpan ke {scaler_path}")
        
        print(f"\n✓ Semua data preprocessing berhasil disimpan ke folder '{output_dir}'")
        return True
        
    except Exception as e:
        print(f"✗ Error saat menyimpan data: {str(e)}")
        return False


def preprocess_gold_price_data(input_path, output_dir, window_size=60, test_year=2022):
    """
    Fungsi utama untuk menjalankan seluruh pipeline preprocessing
    dan termasuk pembuatan sequences untuk LSTM
    """
    print("="*60)
    print("MULAI PREPROCESSING DATA GOLD PRICE")
    print("="*60)
    
    # Langkah 1: Memuat data
    df = load_data(input_path)
    if df is None:
        return None

    # Langkah 2: Menghapus kolom yang tidak diperlukan
    df = drop_unnecessary_columns(df, columns_to_drop=['Vol.', 'Change %'])

    # Langkah 3: Mengonversi dan mengurutkan tanggal
    df = convert_and_sort_date(df, date_column='Date')

    # Langkah 4: Menangani missing values
    df = handle_missing_values(df)

    # Langkah 5: Mengubah nama kolom
    df = rename_columns(df, rename_dict={'Price': 'Close'})

    # Langkah 6: Mengatur tanggal sebagai index
    df = set_date_as_index(df, date_column='Date')

    # Langkah 7: Membersihkan kolom harga
    df = clean_price_columns(df, price_columns=['Close', 'Open', 'High', 'Low'])
    
    # Langkah 8: Melakukan scaling data
    df_scaled, scaler = scale_data(df, feature_column='Close', test_year=test_year)
    
    # Langkah 9: Membuat sequences untuk LSTM
    print("\n" + "="*60)
    print("MEMBUAT SEQUENCES UNTUK LSTM")
    print("="*60)
    X_train, y_train, X_test, y_test = create_sequences(
        df_scaled, 
        feature_column='Close_scaled', 
        window_size=window_size,
        test_year=test_year
    )

    # Langkah 10: Menyimpan semua data yang sudah diproses
    print("\n" + "="*60)
    print("MENYIMPAN DATA PREPROCESSING")
    print("="*60)
    success = save_preprocessed_data(df_scaled, X_train, y_train, X_test, y_test, scaler, output_dir)
    
    if success:
        print("\n" + "="*60)
        print("PREPROCESSING SELESAI!")
        print("="*60)
        print(f"\nFile yang dihasilkan:")
        print(f"  1. {output_dir}/Gold Price (2013-2023)_preprocessing.csv")
        print(f"  2. {output_dir}/Gold Price (2013-2023)_sequences.npz")
        print(f"  3. {output_dir}/scaler.pkl")
        print("="*60)
        return df_scaled, X_train, y_train, X_test, y_test, scaler
    else:
        return None


# Menjalankan preprocessing jika file ini dieksekusi langsung
if __name__ == "__main__":
    import sys
    
    # Mengatur parameter default
    INPUT_FILE = 'Gold Price (2013-2023)_raw.csv'
    OUTPUT_DIR = 'preprocessing/Gold Price (2013-2023)_preprocessing'
    WINDOW_SIZE = 60
    TEST_YEAR = 2022
    
    # Jika ada argument dari command line, gunakan itu
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    
    # Menjalankan preprocessing
    result = preprocess_gold_price_data(
        INPUT_FILE, 
        OUTPUT_DIR,
        window_size=WINDOW_SIZE,
        test_year=TEST_YEAR
    )

    # Menampilkan info hasil
    if result is not None:
        df_scaled, X_train, y_train, X_test, y_test, scaler = result
        print("\n" + "="*60)
        print("INFORMASI DATA HASIL PREPROCESSING:")
        print("="*60)
        print(f"DataFrame shape: {df_scaled.shape}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("\nTipe data DataFrame:")
        print(df_scaled.dtypes)
        print("\n5 baris pertama DataFrame:")
        print(df_scaled.head())
        print("="*60)
    else:
        sys.exit(1)  # Exit dengan error code jika gagal