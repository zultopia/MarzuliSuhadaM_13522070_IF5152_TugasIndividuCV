# Nama: Marzuli Suhada M
# NIM: 13522070
# Fitur unik: Script utama yang mengintegrasikan semua fitur Computer Vision

import os
import sys
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime

def run_filtering():
    """
    Menjalankan proses Image Filtering
    """
    print("=" * 60)
    print("MEMULAI PROSES IMAGE FILTERING")
    print("=" * 60)
    
    try:
        os.chdir('01_filtering')
        result = subprocess.run([sys.executable, 'filtering.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Image Filtering berhasil!")
            print(result.stdout)
        else:
            print("❌ Error dalam Image Filtering:")
            print(result.stderr)
        
        os.chdir('..')
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error menjalankan Image Filtering: {e}")
        os.chdir('..')
        return False

def run_edge_detection():
    """
    Menjalankan proses Edge Detection
    """
    print("=" * 60)
    print("MEMULAI PROSES EDGE DETECTION")
    print("=" * 60)
    
    try:
        os.chdir('02_edge')
        result = subprocess.run([sys.executable, 'edge_detection.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Edge Detection berhasil!")
            print(result.stdout)
        else:
            print("❌ Error dalam Edge Detection:")
            print(result.stderr)
        
        os.chdir('..')
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error menjalankan Edge Detection: {e}")
        os.chdir('..')
        return False

def run_feature_points():
    """
    Menjalankan proses Feature Points Detection
    """
    print("=" * 60)
    print("MEMULAI PROSES FEATURE POINTS DETECTION")
    print("=" * 60)
    
    try:
        os.chdir('03_featurepoints')
        result = subprocess.run([sys.executable, 'feature_points.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Feature Points Detection berhasil!")
            print(result.stdout)
        else:
            print("❌ Error dalam Feature Points Detection:")
            print(result.stderr)
        
        os.chdir('..')
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error menjalankan Feature Points Detection: {e}")
        os.chdir('..')
        return False

def run_geometry_calibration():
    """
    Menjalankan proses Camera Geometry & Calibration
    """
    print("=" * 60)
    print("MEMULAI PROSES CAMERA GEOMETRY & CALIBRATION")
    print("=" * 60)
    
    try:
        os.chdir('04_geometry')
        result = subprocess.run([sys.executable, 'geometry_calibration.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Camera Geometry & Calibration berhasil!")
            print(result.stdout)
        else:
            print("❌ Error dalam Camera Geometry & Calibration:")
            print(result.stderr)
        
        os.chdir('..')
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error menjalankan Camera Geometry & Calibration: {e}")
        os.chdir('..')
        return False

def create_summary_report():
    """
    Membuat laporan ringkasan hasil
    """
    print("=" * 60)
    print("MEMBUAT LAPORAN RINGKASAN")
    print("=" * 60)
    
    summary = f"""
# LAPORAN RINGKASAN TUGAS COMPUTER VISION
## Marzuli Suhada M - 13522070

### Tanggal Eksekusi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Status Eksekusi:
- Image Filtering: {'✅ Berhasil' if os.path.exists('01_filtering/output') else '❌ Gagal'}
- Edge Detection: {'✅ Berhasil' if os.path.exists('02_edge/output') else '❌ Gagal'}
- Feature Points: {'✅ Berhasil' if os.path.exists('03_featurepoints/output') else '❌ Gagal'}
- Geometry & Calibration: {'✅ Berhasil' if os.path.exists('04_geometry/output') else '❌ Gagal'}

### Output yang Dihasilkan:
"""
    
    # Cek output di setiap folder
    for folder in ['01_filtering', '02_edge', '03_featurepoints', '04_geometry']:
        if os.path.exists(f'{folder}/output'):
            files = os.listdir(f'{folder}/output')
            summary += f"\n#### {folder.upper()}:\n"
            for file in files[:5]:  # Maksimal 5 file per folder
                summary += f"- {file}\n"
            if len(files) > 5:
                summary += f"- ... dan {len(files) - 5} file lainnya\n"
    
    summary += f"""
### Catatan:
- Semua script menggunakan gambar standar dari scikit-image
- Output tersimpan dalam folder 'output' di setiap modul
- Tabel analisis tersimpan dalam format CSV
- Gambar hasil tersimpan dalam format PNG dengan resolusi tinggi

### Fitur Unik yang Diimplementasikan:
1. Analisis parameter otomatis dengan visualisasi komprehensif
2. Export hasil dalam format yang mudah dianalisis (CSV, PNG)
3. Dokumentasi lengkap setiap parameter yang digunakan
4. Implementasi modular dengan fungsi terpisah untuk setiap operasi
5. Analisis statistik dan komparasi hasil

---
*Dibuat oleh: Marzuli Suhada M (13522070)*
*Mata Kuliah: IF5152 Computer Vision*
"""
    
    with open('SUMMARY_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("✅ Laporan ringkasan berhasil dibuat: SUMMARY_REPORT.md")

def main():
    """
    Fungsi utama yang menjalankan semua proses Computer Vision
    """
    print("🚀 MEMULAI APLIKASI COMPUTER VISION INTEGRATIF")
    print("👤 Marzuli Suhada M - 13522070")
    print("📚 IF5152 Computer Vision")
    print("=" * 60)
    
    # Dependencies
    try:
        import numpy as np
        import cv2
        from skimage import data
        import matplotlib.pyplot as plt
        import pandas as pd
        print("✅ Semua dependencies tersedia")
    except ImportError as e:
        print(f"❌ Dependencies tidak lengkap: {e}")
        print("Silakan install: pip install numpy opencv-python scikit-image matplotlib pandas")
        return
    
    # Jalankan semua proses
    results = []
    
    # 1. Image Filtering
    results.append(("Image Filtering", run_filtering()))
    
    # 2. Edge Detection
    results.append(("Edge Detection", run_edge_detection()))
    
    # 3. Feature Points Detection
    results.append(("Feature Points Detection", run_feature_points()))
    
    # 4. Camera Geometry & Calibration
    results.append(("Camera Geometry & Calibration", run_geometry_calibration()))

    create_summary_report()
    
    # Hasil akhir
    print("=" * 60)
    print("HASIL AKHIR EKSEKUSI")
    print("=" * 60)
    
    success_count = 0
    for process_name, success in results:
        status = "✅ BERHASIL" if success else "❌ GAGAL"
        print(f"{process_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n📊 Ringkasan: {success_count}/{len(results)} proses berhasil")
    
    if success_count == len(results):
        print("🎉 Semua proses berhasil dijalankan!")
        print("📁 Periksa folder 'output' di setiap modul untuk melihat hasil")
    else:
        print("⚠️  Beberapa proses gagal. Periksa error message di atas")
    
    print("\n📋 Laporan lengkap tersimpan di: SUMMARY_REPORT.md")
    print("=" * 60)

if __name__ == "__main__":
    main()

