# Nama: Marzuli Suhada M
# NIM: 13522070
# Fitur unik: Implementasi filtering dengan analisis parameter otomatis dan visualisasi komparatif

import numpy as np
import cv2
from skimage import data, filters
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_standard_images():
    """
    Memuat gambar standar dari scikit-image
    """
    images = {}
    images_color = {}
    
    # Gambar standar yang wajib digunakan
    images['cameraman'] = data.camera()
    images_color['cameraman'] = cv2.cvtColor(data.camera(), cv2.COLOR_GRAY2RGB)
    
    images['coin'] = data.coins()
    images_color['coin'] = cv2.cvtColor(data.coins(), cv2.COLOR_GRAY2RGB)
    
    images['checkerboard'] = data.checkerboard()
    images_color['checkerboard'] = cv2.cvtColor(data.checkerboard(), cv2.COLOR_GRAY2RGB)
    
    images['astronaut'] = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY)
    images_color['astronaut'] = data.astronaut() 
    
    return images, images_color

def load_personal_images():
    """
    Memuat gambar pribadi dari folder gambar_pribadi
    """
    images = {}
    images_color = {}
    
    # Path ke folder gambar pribadi (dari folder output ke folder gambar_pribadi)
    personal_dir = '../../gambar_pribadi'
    
    # Daftar gambar pribadi
    image_files = ['azul.png', 'madam_eva.png']
    
    for img_file in image_files:
        img_path = os.path.join(personal_dir, img_file)
        if os.path.exists(img_path):
            img_color = cv2.imread(img_path)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            
            # Konversi ke grayscale
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
            
            # Simpan dengan nama tanpa ekstensi
            img_name = os.path.splitext(img_file)[0]
            images[img_name] = img_gray
            images_color[img_name] = img_color
        else:
            print(f"Warning: {img_path} tidak ditemukan")
    
    return images, images_color

def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Menerapkan Gaussian filter pada gambar
    
    Args:
        image: Input image
        kernel_size: Ukuran kernel (harus ganjil)
        sigma: Standar deviasi Gaussian
    
    Returns:
        Filtered image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_median_filter(image, kernel_size=5):
    """
    Menerapkan Median filter pada gambar
    
    Args:
        image: Input image
        kernel_size: Ukuran kernel (harus ganjil)
    
    Returns:
        Filtered image
    """
    return cv2.medianBlur(image, kernel_size)

def apply_sobel_filter(image, direction='both'):
    """
    Menerapkan Sobel filter pada gambar
    
    Args:
        image: Input image
        direction: 'x', 'y', atau 'both'
    
    Returns:
        Filtered image
    """
    if direction == 'x':
        return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    elif direction == 'y':
        return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    else:  # both
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

def analyze_filter_parameters():
    """
    Menganalisis efek parameter filter yang berbeda
    """
    # Parameter untuk Gaussian filter
    gaussian_params = [
        {'kernel_size': 3, 'sigma': 0.5},
        {'kernel_size': 5, 'sigma': 1.0},
        {'kernel_size': 7, 'sigma': 1.5},
        {'kernel_size': 9, 'sigma': 2.0}
    ]
    
    # Parameter untuk Median filter
    median_params = [
        {'kernel_size': 3},
        {'kernel_size': 5},
        {'kernel_size': 7},
        {'kernel_size': 9}
    ]
    
    return gaussian_params, median_params

def save_comparison_images(original, filtered, filter_name, image_name, params, original_color=None):
    """
    Menyimpan gambar perbandingan sebelum dan sesudah filtering
    
    Args:
        original: Original grayscale image
        filtered: Filtered image
        filter_name: Nama filter yang digunakan
        image_name: Nama gambar
        params: Parameter filter
        original_color: Original color image (optional)
    """
    # Jika ada gambar berwarna, tampilkan 3 gambar: color, grayscale, filtered
    if original_color is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Tampilkan gambar original - cek apakah benar-benar berwarna atau grayscale
        if len(original_color.shape) == 3 and original_color.shape[2] == 3:
            # Cek apakah ketiga channel identik (pseudo-color dari grayscale)
            if np.allclose(original_color[:,:,0], original_color[:,:,1]) and \
               np.allclose(original_color[:,:,1], original_color[:,:,2]):
                # Pseudo-color dari grayscale, tampilkan dengan colormap default
                axes[0].imshow(original)
                axes[0].set_title(f'Original - {image_name}')
            else:
                # True color (RGB asli)
                axes[0].imshow(original_color)
                axes[0].set_title(f'Original - {image_name}')
        else:
            # Grayscale, tampilkan dengan colormap default
            axes[0].imshow(original)
            axes[0].set_title(f'Original - {image_name}')
        axes[0].axis('off')
        
        axes[1].imshow(original, cmap='gray')
        axes[1].set_title(f'Original (Grayscale) - {image_name}')
        axes[1].axis('off')
        
        axes[2].imshow(filtered, cmap='gray')
        axes[2].set_title(f'{filter_name} Filtered - {image_name}')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title(f'Original - {image_name}')
        axes[0].axis('off')
        
        axes[1].imshow(filtered, cmap='gray')
        axes[1].set_title(f'{filter_name} Filtered - {image_name}')
        axes[1].axis('off')
    
    plt.tight_layout()
    filename = f'{image_name}_{filter_name.lower().replace(" ", "_")}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_parameter_table(results):
    """
    Membuat tabel parameter yang digunakan dalam filtering
    """
    df = pd.DataFrame(results)
    df.to_csv('filtering_parameters.csv', index=False)
    return df

def main():
    """
    Fungsi utama untuk menjalankan semua operasi filtering
    """
    print("Memulai proses Image Filtering...")
    
    # Membuat folder output jika belum ada
    os.makedirs('output', exist_ok=True)
    os.chdir('output')
    
    # Memuat gambar standar dan pribadi
    images_std, images_color_std = load_standard_images()
    images_personal, images_color_personal = load_personal_images()
    
    # Gabungkan gambar standar dan pribadi
    images = {**images_std, **images_personal}
    images_color = {**images_color_std, **images_color_personal}
    
    # Inisialisasi hasil
    results = []
    
    # Parameter untuk analisis
    gaussian_params, median_params = analyze_filter_parameters()
    
    print("Memproses gambar standar dan pribadi...")
    
    for img_name, img_data in images.items():
        print(f"Memproses {img_name}...")
        
        # Dapatkan gambar berwarna
        img_color = images_color.get(img_name, None)
        
        # Konversi ke grayscale jika diperlukan
        if len(img_data.shape) == 3:
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_data
        
        # Tentukan kategori gambar (standar atau pribadi)
        img_category = 'Pribadi' if img_name in images_personal else 'Standar'
        
        # 1. Gaussian Filter dengan parameter berbeda
        for params in gaussian_params:
            filtered = apply_gaussian_filter(img_gray, 
                                          params['kernel_size'], 
                                          params['sigma'])
            
            # Simpan gambar perbandingan
            filename = save_comparison_images(img_gray, filtered, 
                                            'Gaussian', img_name, params, img_color)
            
            # Simpan hasil ke tabel
            results.append({
                'Image': img_name,
                'Category': img_category,
                'Filter': 'Gaussian',
                'Kernel_Size': params['kernel_size'],
                'Sigma': params['sigma'],
                'Output_File': filename,
                'Original_Shape': img_gray.shape,
                'Filtered_Shape': filtered.shape
            })
        
        # 2. Median Filter dengan parameter berbeda
        for params in median_params:
            filtered = apply_median_filter(img_gray, params['kernel_size'])
            
            # Simpan gambar perbandingan
            filename = save_comparison_images(img_gray, filtered, 
                                            'Median', img_name, params, img_color)
            
            # Simpan hasil ke tabel
            results.append({
                'Image': img_name,
                'Category': img_category,
                'Filter': 'Median',
                'Kernel_Size': params['kernel_size'],
                'Sigma': 'N/A',
                'Output_File': filename,
                'Original_Shape': img_gray.shape,
                'Filtered_Shape': filtered.shape
            })
        
        # 3. Sobel Filter
        sobel_filtered = apply_sobel_filter(img_gray, 'both')
        
        # Normalisasi untuk visualisasi
        sobel_normalized = np.uint8(255 * sobel_filtered / np.max(sobel_filtered))
        
        # Simpan gambar perbandingan
        filename = save_comparison_images(img_gray, sobel_normalized, 
                                        'Sobel', img_name, {}, img_color)
        
        # Simpan hasil ke tabel
        results.append({
            'Image': img_name,
            'Category': img_category,
            'Filter': 'Sobel',
            'Kernel_Size': 3,
            'Sigma': 'N/A',
            'Output_File': filename,
            'Original_Shape': img_gray.shape,
            'Filtered_Shape': sobel_filtered.shape
        })
    
    # Membuat tabel parameter
    print("Membuat tabel parameter...")
    param_table = create_parameter_table(results)
    print("\nTabel Parameter Filtering:")
    print(param_table.to_string(index=False))
    
    # Analisis statistik
    print("\nAnalisis Statistik:")
    print(f"Total gambar yang diproses: {len(images)}")
    print(f"Total operasi filtering: {len(results)}")
    print(f"Filter yang digunakan: {param_table['Filter'].unique()}")
    
    print("\nProses Image Filtering selesai!")
    print("Output tersimpan di folder 'output/'")

if __name__ == "__main__":
    main()

