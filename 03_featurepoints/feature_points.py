# Nama: Marzuli Suhada M
# NIM: 13522070
# Fitur unik: Implementasi feature detection dengan analisis statistik komprehensif

import numpy as np
import cv2
from skimage import data, feature
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_standard_images():
    """Memuat gambar standar dari scikit-image"""
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
    """Memuat gambar pribadi dari folder gambar_pribadi"""
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

def detect_harris_corners(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """Deteksi corner menggunakan Harris corner detection"""
    img_float = np.float32(image)
    corners = cv2.cornerHarris(img_float, block_size, ksize, k)
    corners_thresh = cv2.dilate(corners, None)
    
    corner_coords = np.where(corners_thresh > threshold * corners_thresh.max())
    corner_points = np.column_stack((corner_coords[1], corner_coords[0]))
    corner_responses = corners_thresh[corner_coords]
    
    return corner_points, corner_responses, corners_thresh

def detect_sift_features(image, n_features=0):
    """Deteksi feature menggunakan SIFT"""
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if keypoints:
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        responses = np.array([kp.response for kp in keypoints])
        scales = np.array([kp.size for kp in keypoints])
        angles = np.array([kp.angle for kp in keypoints])
    else:
        points = np.array([])
        responses = np.array([])
        scales = np.array([])
        angles = np.array([])
    
    return keypoints, descriptors, points, responses, scales, angles

def detect_fast_features(image, threshold=10, non_max_suppression=True):
    """Deteksi feature menggunakan FAST"""
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=non_max_suppression)
    keypoints = fast.detect(image, None)
    
    if keypoints:
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        responses = np.array([kp.response for kp in keypoints])
    else:
        points = np.array([])
        responses = np.array([])
    
    return keypoints, points, responses

def analyze_feature_parameters(image, method='harris'):
    """Menganalisis efek parameter yang berbeda pada feature detection"""
    results = {}
    
    if method == 'harris':
        thresholds = [0.01, 0.05, 0.1, 0.2]
        block_sizes = [2, 3, 5]
        
        for block_size in block_sizes:
            for threshold in thresholds:
                points, responses, corner_map = detect_harris_corners(image, block_size=block_size, threshold=threshold)
                
                param_name = f"block_{block_size}_thresh_{threshold}"
                results[param_name] = {
                    'points': points,
                    'responses': responses,
                    'corner_map': corner_map,
                    'num_features': len(points),
                    'block_size': block_size,
                    'threshold': threshold,
                    'mean_response': np.mean(responses) if len(responses) > 0 else 0,
                    'max_response': np.max(responses) if len(responses) > 0 else 0
                }
    
    elif method == 'sift':
        n_features_list = [0, 100, 500, 1000]
        
        for n_features in n_features_list:
            keypoints, descriptors, points, responses, scales, angles = detect_sift_features(image, n_features=n_features)
            
            param_name = f"nfeatures_{n_features}"
            results[param_name] = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'points': points,
                'responses': responses,
                'scales': scales,
                'angles': angles,
                'num_features': len(points),
                'n_features': n_features,
                'mean_response': np.mean(responses) if len(responses) > 0 else 0,
                'mean_scale': np.mean(scales) if len(scales) > 0 else 0
            }
    
    elif method == 'fast':
        thresholds = [5, 10, 20, 30]
        
        for threshold in thresholds:
            keypoints, points, responses = detect_fast_features(image, threshold=threshold)
            
            param_name = f"thresh_{threshold}"
            results[param_name] = {
                'keypoints': keypoints,
                'points': points,
                'responses': responses,
                'num_features': len(points),
                'threshold': threshold,
                'mean_response': np.mean(responses) if len(responses) > 0 else 0,
                'max_response': np.max(responses) if len(responses) > 0 else 0
            }
    
    return results

def visualize_feature_points(image, feature_results, method, image_name, image_color=None):
    """Visualisasi feature points pada gambar"""
    n_results = len(feature_results)
    
    if image_color is not None:
        n_cols = min(3, n_results + 2)
        n_rows = (n_results + 2 + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() if n_results + 2 > 1 else [axes]
        
        # Gambar berwarna (atau colorized jika grayscale)
        if len(image_color.shape) == 3 and image_color.shape[2] == 3:
            # Cek apakah ketiga channel identik (pseudo-color)
            if np.allclose(image_color[:,:,0], image_color[:,:,1]) and \
               np.allclose(image_color[:,:,1], image_color[:,:,2]):
                # Pseudo-color dari grayscale, tampilkan dengan colormap
                axes[0].imshow(image, cmap='viridis')
                axes[0].set_title(f'Original (Colorized) - {image_name}')
            else:
                # True color
                axes[0].imshow(image_color)
                axes[0].set_title(f'Original (Color) - {image_name}')
        else:
            axes[0].imshow(image, cmap='viridis')
            axes[0].set_title(f'Original (Colorized) - {image_name}')
        axes[0].axis('off')
        
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title(f'Original (Grayscale) - {image_name}')
        axes[1].axis('off')
        
        start_idx = 2
    else:
        n_cols = 2
        n_rows = (n_results + 1 + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten() if n_results > 1 else [axes]
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Original - {image_name}')
        axes[0].axis('off')
        
        start_idx = 1
    
    for i, (param_name, result) in enumerate(feature_results.items()):
        if i + start_idx < len(axes):
            axes[i + start_idx].imshow(image, cmap='gray')
            
            if method == 'harris':
                points = result['points']
                responses = result['responses']
                if len(points) > 0:
                    scatter = axes[i + start_idx].scatter(points[:, 0], points[:, 1], 
                                                c=responses, cmap='hot', 
                                                s=20, alpha=0.7)
                    plt.colorbar(scatter, ax=axes[i + start_idx])
            
            elif method == 'sift':
                keypoints = result['keypoints']
                if keypoints:
                    for kp in keypoints:
                        axes[i + start_idx].plot(kp.pt[0], kp.pt[1], 'r+', markersize=kp.size/10)
            
            elif method == 'fast':
                points = result['points']
                responses = result['responses']
                if len(points) > 0:
                    scatter = axes[i + start_idx].scatter(points[:, 0], points[:, 1], 
                                                c=responses, cmap='viridis', 
                                                s=20, alpha=0.7)
                    plt.colorbar(scatter, ax=axes[i + start_idx])
            
            axes[i + start_idx].set_title(f'{method.upper()} - {param_name}\nFeatures: {result["num_features"]}')
            axes[i + start_idx].axis('off')
    
    total_used = start_idx + n_results
    for i in range(total_used, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    filename = f'{image_name}_{method}_features.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_feature_statistics_table(all_results):
    """Membuat tabel statistik feature detection"""
    df = pd.DataFrame(all_results)
    df.to_csv('feature_points_statistics.csv', index=False)
    return df

def main():
    """Fungsi utama untuk menjalankan semua operasi feature detection"""
    print("Memulai proses Feature Points Detection...")
    
    os.makedirs('output', exist_ok=True)
    os.chdir('output')
    
    images_std, images_color_std = load_standard_images()
    images_personal, images_color_personal = load_personal_images()
    
    images = {**images_std, **images_personal}
    images_color = {**images_color_std, **images_color_personal}
    
    all_results = []
    
    print("Memproses gambar standar dan pribadi...")
    
    for img_name, img_data in images.items():
        print(f"Memproses {img_name}...")
        
        img_color = images_color.get(img_name, None)
        img_category = 'Pribadi' if img_name in images_personal else 'Standar'
        
        if len(img_data.shape) == 3:
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_data
        
        print(f"  - Harris corner detection untuk {img_name}")
        harris_results = analyze_feature_parameters(img_gray, 'harris')
        filename = visualize_feature_points(img_gray, harris_results, 'harris', img_name, img_color)
        
        for param_name, result in harris_results.items():
            all_results.append({
                'Image': img_name,
                'Category': img_category,
                'Method': 'Harris',
                'Parameters': param_name,
                'Num_Features': result['num_features'],
                'Mean_Response': result['mean_response'],
                'Max_Response': result['max_response'],
                'Block_Size': result['block_size'],
                'Threshold': result['threshold'],
                'Output_File': filename,
                'Image_Size': img_gray.shape
            })
        
        print(f"  - SIFT feature detection untuk {img_name}")
        sift_results = analyze_feature_parameters(img_gray, 'sift')
        filename = visualize_feature_points(img_gray, sift_results, 'sift', img_name, img_color)
        
        for param_name, result in sift_results.items():
            all_results.append({
                'Image': img_name,
                'Category': img_category,
                'Method': 'SIFT',
                'Parameters': param_name,
                'Num_Features': result['num_features'],
                'Mean_Response': result['mean_response'],
                'Mean_Scale': result['mean_scale'],
                'N_Features': result['n_features'],
                'Output_File': filename,
                'Image_Size': img_gray.shape
            })
        
        print(f"  - FAST feature detection untuk {img_name}")
        fast_results = analyze_feature_parameters(img_gray, 'fast')
        filename = visualize_feature_points(img_gray, fast_results, 'fast', img_name, img_color)
        
        for param_name, result in fast_results.items():
            all_results.append({
                'Image': img_name,
                'Category': img_category,
                'Method': 'FAST',
                'Parameters': param_name,
                'Num_Features': result['num_features'],
                'Mean_Response': result['mean_response'],
                'Max_Response': result['max_response'],
                'Threshold': result['threshold'],
                'Output_File': filename,
                'Image_Size': img_gray.shape
            })
    
    print("Membuat tabel statistik...")
    stats_table = create_feature_statistics_table(all_results)
    print("\nTabel Statistik Feature Points:")
    print(stats_table.to_string(index=False))
    
    print("\nAnalisis Statistik:")
    print(f"Total gambar yang diproses: {len(images)}")
    print(f"Total operasi feature detection: {len(all_results)}")
    print(f"Metode yang digunakan: {stats_table['Method'].unique()}")
    
    for method in stats_table['Method'].unique():
        method_data = stats_table[stats_table['Method'] == method]
        print(f"\n{method}:")
        print(f"  - Rata-rata jumlah features: {method_data['Num_Features'].mean():.1f}")
        print(f"  - Maksimal features: {method_data['Num_Features'].max()}")
        print(f"  - Minimal features: {method_data['Num_Features'].min()}")
        if 'Mean_Response' in method_data.columns:
            print(f"  - Rata-rata response: {method_data['Mean_Response'].mean():.3f}")
    
    print("\nProses Feature Points Detection selesai!")
    print("Output tersimpan di folder 'output/'")

if __name__ == "__main__":
    main()
