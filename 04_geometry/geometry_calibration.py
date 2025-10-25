# Nama: Marzuli Suhada M
# NIM: 13522070
# Fitur unik: Implementasi camera geometry dengan kalibrasi checkerboard dan analisis matrix parameter

import numpy as np
import cv2
from skimage import data
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

def create_synthetic_checkerboard(rows=6, cols=8, square_size=50):
    """Membuat checkerboard sintetik untuk kalibrasi"""
    img_width = cols * square_size
    img_height = rows * square_size
    checkerboard = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                checkerboard[i*square_size:(i+1)*square_size, 
                           j*square_size:(j+1)*square_size] = 255
    
    world_points = np.zeros((rows * cols, 3), np.float32)
    world_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    world_points *= square_size
    
    return checkerboard, world_points

def detect_checkerboard_corners(image, pattern_size=(7, 9)):
    """Deteksi corner checkerboard"""
    ret, corners = cv2.findChessboardCorners(image, pattern_size, None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    
    return ret, corners

def calibrate_camera(world_points, image_points_list, image_size):
    """Kalibrasi kamera menggunakan multiple views"""
    obj_points = [world_points] * len(image_points_list)
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, image_points_list, image_size, None, None)
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

def apply_homography_transform(image, src_points, dst_points):
    """Menerapkan transformasi homography"""
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    h, w = image.shape[:2]
    transformed = cv2.warpPerspective(image, H, (w, h))
    
    return transformed, H

def apply_affine_transform(image, src_points, dst_points):
    """Menerapkan transformasi affine"""
    M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
    
    h, w = image.shape[:2]
    transformed = cv2.warpAffine(image, M, (w, h))
    
    return transformed, M

def analyze_camera_parameters(camera_matrix, dist_coeffs):
    """Menganalisis parameter kamera"""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    focal_length = (fx + fy) / 2
    principal_point = (cx, cy)
    aspect_ratio = fx / fy
    skew = camera_matrix[0, 1]
    
    return {
        'focal_length_x': fx,
        'focal_length_y': fy,
        'focal_length_avg': focal_length,
        'principal_point_x': cx,
        'principal_point_y': cy,
        'aspect_ratio': aspect_ratio,
        'skew': skew,
        'distortion_coeffs': dist_coeffs.flatten()
    }

def visualize_calibration_results(images, corners_list, camera_matrix, dist_coeffs):
    """Visualisasi hasil kalibrasi"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (img, corners) in enumerate(zip(images, corners_list)):
        if i < 4:
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
            
            axes[i].imshow(img, cmap='gray')
            if corners is not None:
                corners_2d = corners.reshape(-1, 2)
                axes[i].plot(corners_2d[:, 0], corners_2d[:, 1], 'r+', markersize=5)
            axes[i].set_title(f'Original Image {i+1}')
            axes[i].axis('off')
    
    plt.tight_layout()
    filename = 'calibration_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def save_transformation_comparison(original, original_color, transformed, transform_name, image_name):
    """Menyimpan perbandingan transformasi dengan gambar asli berwarna"""
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
    
    axes[2].imshow(transformed, cmap='gray')
    axes[2].set_title(f'{transform_name} Transform - {image_name}')
    axes[2].axis('off')
    
    plt.tight_layout()
    filename = f'{image_name}_{transform_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_geometry_analysis_table(results):
    """Membuat tabel analisis geometry dan kalibrasi"""
    df = pd.DataFrame(results)
    df.to_csv('geometry_calibration_analysis.csv', index=False)
    return df

def main():
    """Fungsi utama untuk menjalankan semua operasi geometry dan kalibrasi"""
    print("Memulai proses Camera Geometry & Calibration...")
    
    os.makedirs('output', exist_ok=True)
    os.chdir('output')
    
    all_results = []
    
    print("Membuat checkerboard sintetik untuk kalibrasi...")
    checkerboard_img, world_points = create_synthetic_checkerboard(6, 8, 50)
    cv2.imwrite('synthetic_checkerboard.png', checkerboard_img)
    
    ret, corners = detect_checkerboard_corners(checkerboard_img, (7, 9))
    
    if ret:
        print("Berhasil mendeteksi corners checkerboard sintetik")
        
        image_points_list = []
        images_list = []
        
        image_points_list.append(corners)
        images_list.append(checkerboard_img)
        
        M_rot = cv2.getRotationMatrix2D((checkerboard_img.shape[1]//2, checkerboard_img.shape[0]//2), 15, 1)
        rotated_img = cv2.warpAffine(checkerboard_img, M_rot, checkerboard_img.shape[::-1])
        ret2, corners2 = detect_checkerboard_corners(rotated_img, (7, 9))
        if ret2:
            image_points_list.append(corners2)
            images_list.append(rotated_img)
        
        scaled_img = cv2.resize(checkerboard_img, None, fx=0.8, fy=0.8)
        ret3, corners3 = detect_checkerboard_corners(scaled_img, (7, 9))
        if ret3:
            image_points_list.append(corners3)
            images_list.append(scaled_img)
        
        if len(image_points_list) >= 2:
            ret_cal, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
                world_points, image_points_list, checkerboard_img.shape[::-1])
            
            if ret_cal:
                print("Kalibrasi kamera berhasil!")
                
                cam_params = analyze_camera_parameters(camera_matrix, dist_coeffs)
                
                all_results.append({
                    'Experiment': 'Camera_Calibration',
                    'Method': 'Checkerboard_Synthetic',
                    'Num_Views': len(image_points_list),
                    'Focal_Length_X': cam_params['focal_length_x'],
                    'Focal_Length_Y': cam_params['focal_length_y'],
                    'Principal_Point_X': cam_params['principal_point_x'],
                    'Principal_Point_Y': cam_params['principal_point_y'],
                    'Aspect_Ratio': cam_params['aspect_ratio'],
                    'Skew': cam_params['skew'],
                    'Distortion_Coeffs': str(cam_params['distortion_coeffs']),
                    'Reprojection_Error': ret_cal
                })
                
                filename = visualize_calibration_results(images_list, image_points_list, 
                                                       camera_matrix, dist_coeffs)
                
                print(f"Hasil kalibrasi tersimpan: {filename}")
    
    print("Memproses gambar standar dan pribadi untuk transformasi geometri...")
    
    images_std, images_color_std = load_standard_images()
    images_personal, images_color_personal = load_personal_images()
    
    images = {**images_std, **images_personal}
    images_color = {**images_color_std, **images_color_personal}
    
    for img_name, img_data in images.items():
        print(f"Memproses {img_name}...")
        
        img_color = images_color.get(img_name, None)
        img_category = 'Pribadi' if img_name in images_personal else 'Standar'
        
        if len(img_data.shape) == 3:
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_data
        
        h, w = img_gray.shape
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([[50, 50], [w-50, 100], [w-100, h-50], [100, h-100]])
        
        homography_img, H = apply_homography_transform(img_gray, src_points, dst_points)
        
        if img_color is not None:
            vis_filename = save_transformation_comparison(img_gray, img_color, homography_img, 
                                                        'Homography', img_name)
        else:
            cv2.imwrite(f'{img_name}_homography.png', homography_img)
            vis_filename = f'{img_name}_homography.png'
        
        np.savetxt(f'{img_name}_homography_matrix.txt', H, fmt='%.6f')
        
        src_affine = np.float32([[0, 0], [w, 0], [0, h]])
        dst_affine = np.float32([[50, 50], [w-50, 100], [100, h-50]])
        
        affine_img, M = apply_affine_transform(img_gray, src_affine, dst_affine)
        
        if img_color is not None:
            vis_filename_affine = save_transformation_comparison(img_gray, img_color, affine_img, 
                                                                'Affine', img_name)
        else:
            cv2.imwrite(f'{img_name}_affine.png', affine_img)
            vis_filename_affine = f'{img_name}_affine.png'
        
        np.savetxt(f'{img_name}_affine_matrix.txt', M, fmt='%.6f')
        
        all_results.append({
            'Experiment': 'Geometric_Transform',
            'Method': 'Homography',
            'Image': img_name,
            'Category': img_category,
            'Transform_Matrix': f'{img_name}_homography_matrix.txt',
            'Output_Image': vis_filename,
            'Matrix_Type': '3x3_Homography',
            'Image_Size': img_gray.shape
        })
        
        all_results.append({
            'Experiment': 'Geometric_Transform',
            'Method': 'Affine',
            'Image': img_name,
            'Category': img_category,
            'Transform_Matrix': f'{img_name}_affine_matrix.txt',
            'Output_Image': vis_filename_affine,
            'Matrix_Type': '2x3_Affine',
            'Image_Size': img_gray.shape
        })
    
    print("Membuat tabel analisis...")
    analysis_table = create_geometry_analysis_table(all_results)
    print("\nTabel Analisis Geometry & Calibration:")
    print(analysis_table.to_string(index=False))
    
    print("\nAnalisis Statistik:")
    print(f"Total eksperimen: {len(all_results)}")
    print(f"Jenis eksperimen: {analysis_table['Experiment'].unique()}")
    print(f"Metode yang digunakan: {analysis_table['Method'].unique()}")
    
    calib_data = analysis_table[analysis_table['Experiment'] == 'Camera_Calibration']
    if not calib_data.empty:
        print("\nParameter Kalibrasi Kamera:")
        print(f"  - Focal Length X: {calib_data['Focal_Length_X'].iloc[0]:.2f}")
        print(f"  - Focal Length Y: {calib_data['Focal_Length_Y'].iloc[0]:.2f}")
        print(f"  - Principal Point: ({calib_data['Principal_Point_X'].iloc[0]:.2f}, {calib_data['Principal_Point_Y'].iloc[0]:.2f})")
        print(f"  - Aspect Ratio: {calib_data['Aspect_Ratio'].iloc[0]:.4f}")
    
    print("\nProses Camera Geometry & Calibration selesai!")
    print("Output tersimpan di folder 'output/'")

if __name__ == "__main__":
    main()
