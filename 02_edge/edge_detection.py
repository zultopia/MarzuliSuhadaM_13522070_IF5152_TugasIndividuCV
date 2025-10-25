# Nama: Marzuli Suhada M
# NIM: 13522070
# Fitur unik: Implementasi edge detection dengan analisis threshold dan sampling yang komprehensif

import numpy as np
import cv2
from skimage import data, filters, feature
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
    
    personal_dir = '../../gambar_pribadi'
    image_files = ['azul.png', 'madam_eva.png']
    
    for img_file in image_files:
        img_path = os.path.join(personal_dir, img_file)
        if os.path.exists(img_path):
            img_color = cv2.imread(img_path)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
            img_name = os.path.splitext(img_file)[0]
            images[img_name] = img_gray
            images_color[img_name] = img_color
        else:
            print(f"Warning: {img_path} tidak ditemukan")
    
    return images, images_color

def apply_sobel_edge_detection(image, threshold=50):
    """
    Menerapkan Sobel edge detection
    
    Args:
        image: Input image
        threshold: Threshold untuk binarisasi
    
    Returns:
        Edge map hasil Sobel
    """
    # Sobel gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalisasi dan threshold
    magnitude_normalized = np.uint8(255 * magnitude / np.max(magnitude))
    edge_map = np.where(magnitude_normalized > threshold, 255, 0).astype(np.uint8)
    
    return edge_map, magnitude_normalized

def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Menerapkan Canny edge detection
    
    Args:
        image: Input image
        low_threshold: Lower threshold untuk hysteresis
        high_threshold: Upper threshold untuk hysteresis
    
    Returns:
        Edge map hasil Canny
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def analyze_threshold_effects(image, method='sobel'):
    """
    Menganalisis efek threshold yang berbeda pada edge detection
    
    Args:
        image: Input image
        method: 'sobel' atau 'canny'
    
    Returns:
        Dictionary dengan hasil berbagai threshold
    """
    results = {}
    
    if method == 'sobel':
        thresholds = [20, 50, 80, 120, 150]
        for thresh in thresholds:
            edge_map, magnitude = apply_sobel_edge_detection(image, thresh)
            results[thresh] = {
                'edge_map': edge_map,
                'magnitude': magnitude,
                'edge_pixels': np.sum(edge_map > 0),
                'edge_percentage': (np.sum(edge_map > 0) / edge_map.size) * 100
            }
    
    elif method == 'canny':
        # Kombinasi threshold untuk Canny
        threshold_combinations = [
            (30, 100), (50, 150), (70, 200), (100, 250), (150, 300)
        ]
        for low, high in threshold_combinations:
            edge_map = apply_canny_edge_detection(image, low, high)
            results[f"{low}_{high}"] = {
                'edge_map': edge_map,
                'low_threshold': low,
                'high_threshold': high,
                'edge_pixels': np.sum(edge_map > 0),
                'edge_percentage': (np.sum(edge_map > 0) / edge_map.size) * 100
            }
    
    return results

def apply_sampling_analysis(image, sampling_rates=[1, 2, 4, 8]):
    """
    Menganalisis efek sampling pada edge detection
    
    Args:
        image: Input image
        sampling_rates: List faktor sampling (1 = tidak ada sampling)
    
    Returns:
        Dictionary dengan hasil sampling
    """
    results = {}
    
    for rate in sampling_rates:
        # Downsampling
        height, width = image.shape[:2]
        new_height, new_width = height // rate, width // rate
        
        sampled_image = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
        
        # Edge detection pada gambar yang di-sampling
        sobel_edge, sobel_mag = apply_sobel_edge_detection(sampled_image, 50)
        canny_edge = apply_canny_edge_detection(sampled_image, 50, 150)
        
        # Upsampling kembali ke ukuran asli untuk perbandingan
        sobel_resized = cv2.resize(sobel_edge, (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
        canny_resized = cv2.resize(canny_edge, (width, height), 
                                interpolation=cv2.INTER_NEAREST)
        
        results[rate] = {
            'sampling_rate': rate,
            'sampled_size': (new_height, new_width),
            'sobel_edge': sobel_resized,
            'canny_edge': canny_resized,
            'sobel_pixels': np.sum(sobel_resized > 0),
            'canny_pixels': np.sum(canny_resized > 0)
        }
    
    return results

def save_edge_comparison(original, edge_maps, method, image_name, params, original_color=None):
    """
    Menyimpan gambar perbandingan edge detection
    
    Args:
        original: Original grayscale image
        edge_maps: Dictionary of edge maps dengan berbagai parameter
        method: Metode edge detection
        image_name: Nama gambar
        params: Parameter yang digunakan
        original_color: Original color image (optional)
    """
    n_maps = len(edge_maps)
    
    # Jika ada gambar berwarna, tambahkan kolom untuk color dan grayscale
    if original_color is not None:
        n_cols = min(3, n_maps + 2)  # Maksimal 3 kolom: color, gray, edge
        n_rows = (n_maps + 2 + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() if n_maps + 2 > 1 else [axes]
        
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
        
        # Gambar grayscale
        axes[1].imshow(original, cmap='gray')
        axes[1].set_title(f'Original (Grayscale) - {image_name}')
        axes[1].axis('off')
        
        # Edge maps
        for i, (param_name, edge_map) in enumerate(edge_maps.items()):
            if i + 2 < len(axes):
                axes[i + 2].imshow(edge_map, cmap='gray')
                axes[i + 2].set_title(f'{method} - {param_name}')
                axes[i + 2].axis('off')
        
        # Sembunyikan axes yang tidak digunakan
        for i in range(n_maps + 2, len(axes)):
            axes[i].axis('off')
    else:
        fig, axes = plt.subplots(2, (n_maps + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_maps > 1 else [axes]
        
        # Gambar asli
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title(f'Original - {image_name}')
        axes[0].axis('off')
        
        # Edge maps
        for i, (param_name, edge_map) in enumerate(edge_maps.items()):
            if i + 1 < len(axes):
                axes[i + 1].imshow(edge_map, cmap='gray')
                axes[i + 1].set_title(f'{method} - {param_name}')
                axes[i + 1].axis('off')
        
        # Sembunyikan axes yang tidak digunakan
        for i in range(n_maps + 1, len(axes)):
            axes[i].axis('off')
    
    plt.tight_layout()
    filename = f'{image_name}_{method.lower()}_edge_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_edge_analysis_table(results):
    """
    Membuat tabel analisis edge detection
    """
    df = pd.DataFrame(results)
    df.to_csv('edge_detection_analysis.csv', index=False)
    return df

def main():
    """
    Fungsi utama untuk menjalankan semua operasi edge detection
    """
    print("Memulai proses Edge Detection...")
    
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
    all_results = []
    
    print("Memproses gambar standar dan pribadi...")
    
    for img_name, img_data in images.items():
        print(f"Memproses {img_name}...")
        
        # Dapatkan gambar berwarna
        img_color = images_color.get(img_name, None)
        
        # Tentukan kategori gambar (standar atau pribadi)
        img_category = 'Pribadi' if img_name in images_personal else 'Standar'
        
        # Konversi ke grayscale jika diperlukan
        if len(img_data.shape) == 3:
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_data
        
        # 1. Analisis Threshold Sobel
        print(f"  - Analisis threshold Sobel untuk {img_name}")
        sobel_results = analyze_threshold_effects(img_gray, 'sobel')
        
        # Simpan perbandingan Sobel
        edge_maps = {f"threshold_{thresh}": result['edge_map'] 
                    for thresh, result in sobel_results.items()}
        filename = save_edge_comparison(img_gray, edge_maps, 'Sobel', img_name, {}, img_color)
        
        # Simpan hasil ke tabel
        for thresh, result in sobel_results.items():
            all_results.append({
                'Image': img_name,
                'Category': img_category,
                'Method': 'Sobel',
                'Threshold': thresh,
                'Edge_Pixels': result['edge_pixels'],
                'Edge_Percentage': result['edge_percentage'],
                'Output_File': filename,
                'Image_Size': img_gray.shape
            })
        
        # 2. Analisis Threshold Canny
        print(f"  - Analisis threshold Canny untuk {img_name}")
        canny_results = analyze_threshold_effects(img_gray, 'canny')
        
        # Simpan perbandingan Canny
        edge_maps = {f"low_{result['low_threshold']}_high_{result['high_threshold']}": 
                    result['edge_map'] for result in canny_results.values()}
        filename = save_edge_comparison(img_gray, edge_maps, 'Canny', img_name, {}, img_color)
        
        # Simpan hasil ke tabel
        for param_name, result in canny_results.items():
            all_results.append({
                'Image': img_name,
                'Category': img_category,
                'Method': 'Canny',
                'Threshold': param_name,
                'Edge_Pixels': result['edge_pixels'],
                'Edge_Percentage': result['edge_percentage'],
                'Output_File': filename,
                'Image_Size': img_gray.shape
            })
        
        # 3. Analisis Sampling
        print(f"  - Analisis sampling untuk {img_name}")
        sampling_results = apply_sampling_analysis(img_gray)
        
        # Simpan perbandingan sampling
        edge_maps = {f"rate_{rate}": result['sobel_edge'] 
                    for rate, result in sampling_results.items()}
        filename = save_edge_comparison(img_gray, edge_maps, 'Sampling_Sobel', img_name, {}, img_color)
        
        # Simpan hasil sampling ke tabel
        for rate, result in sampling_results.items():
            all_results.append({
                'Image': img_name,
                'Category': img_category,
                'Method': 'Sampling_Sobel',
                'Threshold': f"rate_{rate}",
                'Edge_Pixels': result['sobel_pixels'],
                'Edge_Percentage': (result['sobel_pixels'] / img_gray.size) * 100,
                'Output_File': filename,
                'Image_Size': img_gray.shape,
                'Sampled_Size': result['sampled_size']
            })
    
    # Membuat tabel analisis
    print("Membuat tabel analisis...")
    analysis_table = create_edge_analysis_table(all_results)
    print("\nTabel Analisis Edge Detection:")
    print(analysis_table.to_string(index=False))
    
    # Analisis statistik
    print("\nAnalisis Statistik:")
    print(f"Total gambar yang diproses: {len(images)}")
    print(f"Total operasi edge detection: {len(all_results)}")
    print(f"Metode yang digunakan: {analysis_table['Method'].unique()}")
    
    # Analisis threshold terbaik
    print("\nAnalisis Threshold:")
    for method in analysis_table['Method'].unique():
        if method != 'Sampling_Sobel':
            method_data = analysis_table[analysis_table['Method'] == method]
            print(f"\n{method}:")
            print(f"  - Rata-rata edge pixels: {method_data['Edge_Pixels'].mean():.0f}")
            print(f"  - Rata-rata edge percentage: {method_data['Edge_Percentage'].mean():.2f}%")
    
    print("\nProses Edge Detection selesai!")
    print("Output tersimpan di folder 'output/'")

if __name__ == "__main__":
    main()

