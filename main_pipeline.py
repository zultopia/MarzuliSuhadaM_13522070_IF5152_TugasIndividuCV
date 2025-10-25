# Nama: Marzuli Suhada M
# NIM: 13522070
# Fitur unik: Pipeline Computer Vision Terintegrasi

import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

class ComputerVisionPipeline:
    """
    Pipeline terintegrasi untuk Computer Vision
    Filtering → Edge Detection → Feature Points → Geometry Calibration
    """
    
    def __init__(self, output_dir='pipeline_output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {
            'filtering': [],
            'edge': [],
            'features': [],
            'geometry': []
        }
    
    def load_images(self):
        """Memuat gambar standar dan pribadi"""
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
        
        # Gambar pribadi
        personal_dir = 'gambar_pribadi'
        for img_file in ['azul.png', 'madam_eva.png']:
            img_path = os.path.join(personal_dir, img_file)
            if os.path.exists(img_path):
                img_color = cv2.imread(img_path)
                img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
                
                img_name = os.path.splitext(img_file)[0]
                images[img_name] = img_gray
                images_color[img_name] = img_color
        
        return images, images_color
    
    def step1_filtering(self, image, method='gaussian', kernel_size=5, sigma=1.5):
        """
        STEP 1: Image Filtering
        Input: Raw image
        Output: Filtered image (noise reduced)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == 'gaussian':
            filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif method == 'median':
            filtered = cv2.medianBlur(image, kernel_size)
        else:
            filtered = image.copy()
        
        return filtered
    
    def step2_edge_detection(self, filtered_image, method='canny', low_thresh=50, high_thresh=150):
        """
        STEP 2: Edge Detection
        Input: Filtered image (from step 1)
        Output: Edge map
        """
        if method == 'canny':
            edges = cv2.Canny(filtered_image, low_thresh, high_thresh)
        elif method == 'sobel':
            sobelx = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(filtered_image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(255 * magnitude / np.max(magnitude))
            edges = np.where(edges > 50, 255, 0).astype(np.uint8)
        else:
            edges = filtered_image.copy()
        
        return edges
    
    def step3_feature_detection(self, edge_image, filtered_image, method='harris', threshold=0.01):
        """
        STEP 3: Feature Point Detection
        Input: Edge map (from step 2) + Filtered image (for context)
        Output: Feature points (corners/keypoints)
        """
        # Gunakan filtered image untuk deteksi fitur, dengan edge sebagai mask/guidance
        edge_mask = (edge_image > 0).astype(np.uint8) * 255
        
        if method == 'harris':
            img_float = np.float32(filtered_image)
            corners = cv2.cornerHarris(img_float, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            
            # Filter corners yang berada di dekat edges
            corners_on_edges = corners * (edge_mask / 255.0)
            
            corner_coords = np.where(corners_on_edges > threshold * corners_on_edges.max())
            feature_points = np.column_stack((corner_coords[1], corner_coords[0]))
            
        elif method == 'sift':
            sift = cv2.SIFT_create(nfeatures=100)
            keypoints, descriptors = sift.detectAndCompute(filtered_image, mask=edge_mask)
            feature_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints]) if keypoints else np.array([])
            
        elif method == 'fast':
            fast = cv2.FastFeatureDetector_create(threshold=10)
            keypoints = fast.detect(filtered_image, mask=edge_mask)
            feature_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints]) if keypoints else np.array([])
        else:
            feature_points = np.array([])
        
        return feature_points
    
    def step4_geometry_calibration(self, feature_points, image_shape):
        """
        STEP 4: Geometry & Calibration
        Input: Feature points (from step 3)
        Output: Transformation matrix
        """
        h, w = image_shape[:2]
        
        if len(feature_points) >= 4:
            # Gunakan feature points untuk estimasi homography
            # Pilih 4 feature points yang tersebar
            if len(feature_points) > 4:
                indices = np.linspace(0, len(feature_points)-1, 4, dtype=int)
                src_points = feature_points[indices].astype(np.float32)
            else:
                src_points = feature_points[:4].astype(np.float32)
            
            # Target points (contoh: perspective transform)
            dst_points = np.float32([
                [w*0.1, h*0.1],
                [w*0.9, h*0.1],
                [w*0.9, h*0.9],
                [w*0.1, h*0.9]
            ])
            
            try:
                H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
                return H, src_points, dst_points
            except:
                return None, src_points, None
        else:
            return None, feature_points, None
    
    def visualize_pipeline(self, image_name, original, filtered, edges, features, 
                          feature_points, transform_matrix, image_color=None):
        """Visualisasi semua tahap pipeline"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original - tampilkan color jika tersedia (astronaut, azul, madam_eva)
        # atau colorized untuk grayscale (cameraman, coin, checkerboard)
        if image_color is not None and len(image_color.shape) == 3:
            # Cek apakah benar-benar RGB atau grayscale yang diconvert
            if not (np.allclose(image_color[:,:,0], image_color[:,:,1]) and 
                    np.allclose(image_color[:,:,1], image_color[:,:,2])):
                # True RGB - tampilkan warna asli
                axes[0, 0].imshow(image_color)
                axes[0, 0].set_title(f'0. Original (RGB)\n{image_name}')
            else:
                # Grayscale converted to RGB - tampilkan colorized
                axes[0, 0].imshow(original, cmap='viridis')
                axes[0, 0].set_title(f'0. Original (Colorized)\n{image_name}')
        else:
            # Tidak ada image_color - tampilkan colorized
            axes[0, 0].imshow(original, cmap='viridis')
            axes[0, 0].set_title(f'0. Original (Colorized)\n{image_name}')
        axes[0, 0].axis('off')
        
        # Step 1: Filtered
        axes[0, 1].imshow(filtered, cmap='gray')
        axes[0, 1].set_title(f'1. Filtered (Gaussian)\nNoise Reduced')
        axes[0, 1].axis('off')
        
        # Step 2: Edges
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title(f'2. Edge Detection (Canny)\nFrom Filtered Image')
        axes[0, 2].axis('off')
        
        # Step 3: Features
        axes[1, 0].imshow(filtered, cmap='gray')
        if len(feature_points) > 0:
            axes[1, 0].scatter(feature_points[:, 0], feature_points[:, 1], 
                             c='red', s=20, alpha=0.7, marker='o')
        axes[1, 0].set_title(f'3. Feature Points (Harris)\n{len(feature_points)} points from edges')
        axes[1, 0].axis('off')
        
        # Step 4: Geometry (with feature points)
        axes[1, 1].imshow(original, cmap='gray')
        if len(feature_points) > 0:
            axes[1, 1].scatter(feature_points[:, 0], feature_points[:, 1], 
                             c='yellow', s=30, alpha=0.8, marker='+')
        axes[1, 1].set_title(f'4. Geometry Transform\nUsing {len(feature_points)} feature points')
        axes[1, 1].axis('off')
        
        # Pipeline Flow Diagram
        axes[1, 2].text(0.5, 0.9, 'PIPELINE FLOW', ha='center', va='top', 
                       fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
        
        pipeline_text = """
Original Image
     ↓
1. FILTERING
   (Gaussian Blur)
   → Reduce Noise
     ↓
2. EDGE DETECTION
   (Canny)
   → Sharp Edges
     ↓
3. FEATURE POINTS
   (Harris on Edges)
   → Key Points
     ↓
4. GEOMETRY
   (Homography)
   → Transform Matrix
"""
        axes[1, 2].text(0.1, 0.75, pipeline_text, ha='left', va='top',
                       fontsize=10, family='monospace', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f'{image_name}_pipeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def run_pipeline(self, image_name, image, image_color=None):
        """Menjalankan full pipeline untuk satu gambar"""
        print(f"\n{'='*60}")
        print(f"Processing: {image_name}")
        print(f"{'='*60}")
        
        if len(image.shape) == 3:
            original = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            original = image.copy()
        
        # STEP 1: Filtering
        print("Step 1: Filtering (Gaussian) → Mengurangi noise...")
        filtered = self.step1_filtering(original, method='gaussian', kernel_size=5, sigma=1.5)
        noise_reduction = np.std(original) - np.std(filtered)
        print(f"  ✓ Noise reduction: {noise_reduction:.2f}")
        
        self.results['filtering'].append({
            'Image': image_name,
            'Method': 'Gaussian',
            'Kernel_Size': 5,
            'Sigma': 1.5,
            'Noise_Reduction': noise_reduction,
            'Original_Std': np.std(original),
            'Filtered_Std': np.std(filtered)
        })
        
        # STEP 2: Edge Detection (dari filtered image)
        print("Step 2: Edge Detection (Canny) → Menggunakan filtered image...")
        edges = self.step2_edge_detection(filtered, method='canny', low_thresh=50, high_thresh=150)
        edge_pixels = np.sum(edges > 0)
        edge_percentage = (edge_pixels / edges.size) * 100
        print(f"  ✓ Edge pixels: {edge_pixels} ({edge_percentage:.2f}%)")
        
        self.results['edge'].append({
            'Image': image_name,
            'Method': 'Canny',
            'Input': 'Filtered Image',
            'Edge_Pixels': edge_pixels,
            'Edge_Percentage': edge_percentage
        })
        
        # STEP 3: Feature Detection (dari edges + filtered)
        print("Step 3: Feature Detection (Harris) → Menggunakan edge map...")
        feature_points = self.step3_feature_detection(edges, filtered, method='harris', threshold=0.01)
        num_features = len(feature_points)
        print(f"  ✓ Feature points detected: {num_features}")
        
        self.results['features'].append({
            'Image': image_name,
            'Method': 'Harris',
            'Input': 'Edge Map + Filtered',
            'Num_Features': num_features,
            'Features_on_Edges': True
        })
        
        # STEP 4: Geometry (dari feature points)
        print("Step 4: Geometry Calibration → Menggunakan feature points...")
        transform_matrix, src_pts, dst_pts = self.step4_geometry_calibration(feature_points, original.shape)
        
        if transform_matrix is not None:
            print(f"  ✓ Transform matrix computed from {len(src_pts)} points")
            matrix_valid = True
        else:
            print(f"  ⚠ Insufficient points for transform (need ≥4, got {num_features})")
            matrix_valid = False
        
        self.results['geometry'].append({
            'Image': image_name,
            'Method': 'Homography',
            'Input': 'Feature Points',
            'Num_Points_Used': len(src_pts) if src_pts is not None else 0,
            'Matrix_Valid': matrix_valid,
            'Matrix_Shape': '3x3' if matrix_valid else 'N/A'
        })
        
        # Visualisasi pipeline
        output_file = self.visualize_pipeline(
            image_name, original, filtered, edges, 
            filtered, feature_points, transform_matrix, image_color
        )
        print(f"  ✓ Pipeline visualization saved: {output_file}")
        
        return {
            'original': original,
            'filtered': filtered,
            'edges': edges,
            'feature_points': feature_points,
            'transform_matrix': transform_matrix
        }
    
    def save_analysis(self):
        """Menyimpan analisis statistik pipeline"""
        output_file = os.path.join(self.output_dir, 'pipeline_analysis.csv')
        
        # Combine all results
        all_data = []
        for stage, results in self.results.items():
            for result in results:
                result['Stage'] = stage
                all_data.append(result)
        
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print("PIPELINE ANALYSIS")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\nAnalysis saved to: {output_file}")
        
        return df

def main():
    print("="*70)
    print("COMPUTER VISION PIPELINE - TERINTEGRASI")
    print("Filtering → Edge Detection → Feature Points → Geometry Calibration")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Inisialisasi pipeline
    pipeline = ComputerVisionPipeline(output_dir='pipeline_output')
    
    # Load images
    print("\nLoading images...")
    images, images_color = pipeline.load_images()
    print(f"✓ Loaded {len(images)} images")
    
    # Proses setiap gambar melalui pipeline
    for img_name, img_data in images.items():
        img_color = images_color.get(img_name, None)
        results = pipeline.run_pipeline(img_name, img_data, img_color)
    
    # Save analysis
    print("\n" + "="*70)
    print("SAVING PIPELINE ANALYSIS...")
    print("="*70)
    df = pipeline.save_analysis()
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(images)}")
    print(f"\nStage Statistics:")
    for stage in ['filtering', 'edge', 'features', 'geometry']:
        stage_data = df[df['Stage'] == stage]
        print(f"\n{stage.upper()}:")
        print(f"  - Total operations: {len(stage_data)}")
        if stage == 'features':
            print(f"  - Avg features detected: {stage_data['Num_Features'].mean():.1f}")
        elif stage == 'edge':
            print(f"  - Avg edge percentage: {stage_data['Edge_Percentage'].mean():.2f}%")
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"✓ Output saved to: {pipeline.output_dir}/")
    print("="*70)

if __name__ == "__main__":
    main()

