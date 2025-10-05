import pandas as pd
import numpy as np
import os


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


def save_preprocessed_data(df, output_path):
    """
    Menyimpan data yang sudah diproses
    
    """
    try:
        # Buat folder jika belum ada
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_csv(output_path)
        print(f"✓ Data preprocessing berhasil disimpan ke {output_path}")
        print(f"  Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"✗ Error saat menyimpan data: {str(e)}")
        return False


def preprocess_gold_price_data(input_path, output_path):
    """
    Fungsi utama untuk menjalankan seluruh pipeline preprocessing
    
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

    # Langkah 8: Menyimpan data yang sudah diproses
    success = save_preprocessed_data(df, output_path)
    
    if success:
        print("="*60)
        print("PREPROCESSING SELESAI!")
        print("="*60)
        return df
    else:
        return None


# Menjalankan preprocessing jika file ini dieksekusi langsung
if __name__ == "__main__":
    import sys
    
    # Default paths
    INPUT_FILE = 'Gold Price (2013-2023)_raw.csv'
    OUTPUT_FILE = 'preprocessing/Gold Price (2013-2023)_preprocessing.csv'
    
    # Jika ada argument dari command line, gunakan itu
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
    
    # Menjalankan preprocessing
    df_processed = preprocess_gold_price_data(INPUT_FILE, OUTPUT_FILE)

    # Menampilkan info hasil
    if df_processed is not None:
        print("\n" + "="*60)
        print("INFORMASI DATA HASIL PREPROCESSING:")
        print("="*60)
        print(f"Shape: {df_processed.shape}")
        print(f"\nTipe data:\n{df_processed.dtypes}")
        print(f"\n5 baris pertama:\n{df_processed.head()}")
    else:
        sys.exit(1)  # Exit dengan error code jika gagal