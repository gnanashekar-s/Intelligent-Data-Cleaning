from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse, HttpResponse
import os
import numpy as np
import pandas as pd
from django.core.paginator import Paginator
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import base64
import zipfile
import shutil
import json

def paginate_images(request):
    page_number = request.GET.get("page", 1)
    df = pd.read_json(request.session.get("image_data", "[]"))
    if df.empty:
        return None
    paginator = Paginator(df.to_dict(orient="records"), 5)
    page_obj = paginator.get_page(page_number)
    return page_obj

def image_preprocessing(request):
    page_obj = paginate_images(request)
    
    # Initialize session variables if they don't exist
    if "image_data" not in request.session:
        request.session["image_data"] = "[]"
    if "preview_images" not in request.session:
        request.session["preview_images"] = []
        
    context = {
        "page_obj": page_obj,
        "preview_images": request.session.get("preview_images", [])
    }
    
    if request.method == "POST":
        action = request.POST.get("action")
        
        # Handle dataset upload
        if action == "upload_dataset":
            if request.FILES.getlist("datasetUpload"):
                uploaded_files = request.FILES.getlist("datasetUpload")
                
                # Create directory for uploaded images
                upload_dir = os.path.join(settings.MEDIA_ROOT, "images", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save preview of first 5 images
                preview_images = []
                for i, img_file in enumerate(uploaded_files[:5]):
                    # Save the file
                    img_path = os.path.join(upload_dir, img_file.name)
                    with open(img_path, "wb+") as destination:
                        for chunk in img_file.chunks():
                            destination.write(chunk)
                    
                    # Add to preview list
                    preview_images.append({
                        "name": img_file.name,
                        "url": f"/media/images/uploads/{img_file.name}",
                        "size": img_file.size
                    })
                
                # Update session with previews
                request.session["preview_images"] = preview_images
                request.session["total_uploaded"] = len(uploaded_files)
                
                context["preview_images"] = preview_images
                context["total_uploaded"] = len(uploaded_files)
                return render(request, 'preprocessing/image_preprocessing.html', context)
                
        # Handle preprocessing
        elif action == "start_preprocessing":
            resize_option = request.POST.get("resize_option", "No Resize")
            color_option = request.POST.get("color_option", "Maintain Original")
            augmentation_option = request.POST.get("augmentation_option", "No Augmentation")
            normalization_option = request.POST.get("normalization_option", "No Normalization")
            noise_reduction_option = request.POST.get("noise_reduction_option", "No Noise Reduction")
            format_option = request.POST.get("format_option", "Maintain Original")
            
            # Process uploaded images
            upload_dir = os.path.join(settings.MEDIA_ROOT, "images", "uploads")
            processed_dir = os.path.join(settings.MEDIA_ROOT, "images", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # Get list of uploaded images
            image_files = []
            for root, dirs, files in os.walk(upload_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_files.append(os.path.join(root, file))
            
            # Process images and collect data
            processed_data = []
            processed_previews = []
            
            for img_path in image_files[:50]:  # Limit to first 50 images for demo
                try:
                    # Get original image info
                    filename = os.path.basename(img_path)
                    orig_img = cv2.imread(img_path)
                    h, w, c = orig_img.shape if orig_img is not None else (0, 0, 0)
                    
                    # Apply preprocessing
                    processed_img = process_image(
                        orig_img, 
                        resize_option, 
                        color_option, 
                        normalization_option,
                        noise_reduction_option
                    )
                    
                    if processed_img is None:
                        continue
                    
                    # Save processed image
                    output_format = get_output_format(format_option, filename)
                    output_filename = f"processed_{filename.split('.')[0]}.{output_format}"
                    output_path = os.path.join(processed_dir, output_filename)
                    
                    cv2.imwrite(output_path, processed_img)
                    
                    # Apply augmentation if selected
                    augmented_paths = []
                    if augmentation_option != "No Augmentation":
                        augmented_paths = apply_augmentation(
                            processed_img,
                            augmentation_option,
                            processed_dir,
                            filename.split('.')[0],
                            output_format
                        )
                    
                    # Get processed image info
                    p_h, p_w, p_c = processed_img.shape
                    
                    # Add to dataset
                    processed_data.append({
                        "filename": filename,
                        "original_size": f"{w}x{h}",
                        "processed_size": f"{p_w}x{p_h}",
                        "color_transform": color_option,
                        "normalization": normalization_option,
                        "noise_reduction": noise_reduction_option,
                        "augmentation": augmentation_option,
                        "format": output_format,
                        "processed_path": f"/media/images/processed/{output_filename}"
                    })
                    
                    # Add to previews (first 5)
                    if len(processed_previews) < 5:
                        processed_previews.append({
                            "name": output_filename,
                            "url": f"/media/images/processed/{output_filename}",
                            "original": f"/media/images/uploads/{filename}"
                        })
                        
                    # Add augmented images to dataset
                    for aug_path in augmented_paths:
                        aug_filename = os.path.basename(aug_path)
                        processed_data.append({
                            "filename": f"aug_{filename}",
                            "original_size": f"{w}x{h}",
                            "processed_size": f"{p_w}x{p_h}",
                            "color_transform": color_option,
                            "normalization": normalization_option,
                            "noise_reduction": noise_reduction_option,
                            "augmentation": f"{augmentation_option} - {aug_filename.split('_')[0]}",
                            "format": output_format,
                            "processed_path": f"/media/images/processed/{aug_filename}"
                        })
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Update session with processed data
            df = pd.DataFrame(processed_data)
            request.session["image_data"] = df.to_json(orient="records")
            request.session["processed_previews"] = processed_previews
            
            # Update context for rendering
            context["processed_previews"] = processed_previews
            context["preprocessing_complete"] = True
            context["total_processed"] = len(processed_data)
            context["page_obj"] = paginate_images(request)
            
            return render(request, 'preprocessing/image_preprocessing.html', context)
            
        # Handle download
        elif action == "download_dataset":
            # Create zip file of processed images
            processed_dir = os.path.join(settings.MEDIA_ROOT, "images", "processed")
            zip_path = os.path.join(settings.MEDIA_ROOT, "processed_images.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(processed_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), 
                                  os.path.relpath(os.path.join(root, file), 
                                                 os.path.join(processed_dir, '..')))
            
            # Create CSV file with metadata
            df = pd.read_json(request.session.get("image_data", "[]"))
            csv_path = os.path.join(settings.MEDIA_ROOT, "image_metadata.csv")
            df.to_csv(csv_path, index=False)
            
            # Add CSV to zip
            with zipfile.ZipFile(zip_path, 'a') as zipf:
                zipf.write(csv_path, "image_metadata.csv")
            
            # Return zip file
            with open(zip_path, 'rb') as f:
                response = HttpResponse(
                    f.read(),
                    content_type='application/zip'
                )
                response['Content-Disposition'] = 'attachment; filename=processed_images.zip'
                return response
    
    return render(request, 'preprocessing/image_preprocessing.html', context)

def process_image(img, resize_option, color_option, normalization_option, noise_reduction_option):
    """Apply selected preprocessing steps to an image"""
    if img is None:
        return None
    
    # Apply resize
    if resize_option != "No Resize":
        if resize_option == "Resize to 224x224":
            img = cv2.resize(img, (224, 224))
        elif resize_option == "Resize to 299x299":
            img = cv2.resize(img, (299, 299))
        elif resize_option == "Custom Size":
            img = cv2.resize(img, (256, 256))  # Default custom size
    
    # Apply color transformation
    if color_option != "Maintain Original":
        if color_option == "Convert to Grayscale":
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels for consistency
        elif color_option == "Normalize Colors":
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        elif color_option == "HSV Conversion":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Apply noise reduction
    if noise_reduction_option != "No Noise Reduction":
        if noise_reduction_option == "Light Smoothing":
            img = cv2.GaussianBlur(img, (3, 3), 0)
        elif noise_reduction_option == "Medium Denoising":
            img = cv2.medianBlur(img, 5)
        elif noise_reduction_option == "Advanced Filtering":
            img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply normalization
    if normalization_option != "No Normalization":
        if normalization_option == "Min-Max Scaling":
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        elif normalization_option == "Z-Score Normalization":
            img = img.astype(np.float32)
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / std
            img = np.clip(img * 64 + 128, 0, 255).astype(np.uint8)
        elif normalization_option == "ImageNet Normalization":
            img = img.astype(np.float32) / 255.0
            img[:,:,0] = (img[:,:,0] - 0.485) / 0.229
            img[:,:,1] = (img[:,:,1] - 0.456) / 0.224
            img[:,:,2] = (img[:,:,2] - 0.406) / 0.225
            img = np.clip(img * 64 + 128, 0, 255).astype(np.uint8)
    
    return img

def apply_augmentation(img, augmentation_option, output_dir, base_filename, output_format):
    """Apply data augmentation techniques to an image"""
    augmented_paths = []
    
    # Light augmentation: flip
    if augmentation_option in ["Light Augmentation", "Medium Augmentation", "Heavy Augmentation"]:
        # Horizontal flip
        flipped = cv2.flip(img, 1)
        flip_path = os.path.join(output_dir, f"flip_{base_filename}.{output_format}")
        cv2.imwrite(flip_path, flipped)
        augmented_paths.append(flip_path)
    
    # Medium augmentation: rotation, brightness
    if augmentation_option in ["Medium Augmentation", "Heavy Augmentation"]:
        # Rotation
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h))
        rotate_path = os.path.join(output_dir, f"rotate_{base_filename}.{output_format}")
        cv2.imwrite(rotate_path, rotated)
        augmented_paths.append(rotate_path)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        bright_path = os.path.join(output_dir, f"bright_{base_filename}.{output_format}")
        cv2.imwrite(bright_path, bright)
        augmented_paths.append(bright_path)
    
    # Heavy augmentation: noise, blur
    if augmentation_option == "Heavy Augmentation":
        # Add noise
        noise = np.copy(img)
        noise_val = np.random.normal(0, 15, noise.shape).astype(np.uint8)
        noise = cv2.add(noise, noise_val)
        noise_path = os.path.join(output_dir, f"noise_{base_filename}.{output_format}")
        cv2.imwrite(noise_path, noise)
        augmented_paths.append(noise_path)
        
        # Random crop
        h, w = img.shape[:2]
        crop_h, crop_w = int(h * 0.8), int(w * 0.8)
        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)
        cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
        cropped = cv2.resize(cropped, (w, h))
        crop_path = os.path.join(output_dir, f"crop_{base_filename}.{output_format}")
        cv2.imwrite(crop_path, cropped)
        augmented_paths.append(crop_path)
    
    return augmented_paths

def get_output_format(format_option, filename):
    """Determine output format based on selection and original filename"""
    if format_option == "Convert to JPEG":
        return "jpg"
    elif format_option == "Convert to PNG":
        return "png"
    elif format_option == "Compress Images":
        return "jpg"  # Use JPEG for compression
    else:
        # Maintain original format
        ext = filename.split('.')[-1].lower()
        return ext if ext in ['jpg', 'jpeg', 'png', 'bmp'] else 'jpg'

def download_processed_images(request):
    """Download all processed images as a zip file"""
    processed_dir = os.path.join(settings.MEDIA_ROOT, "images", "processed")
    zip_path = os.path.join(settings.MEDIA_ROOT, "processed_images.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                          os.path.join(processed_dir, '..')))
    
    with open(zip_path, 'rb') as f:
        response = HttpResponse(
            f.read(),
            content_type='application/zip'
        )
        response['Content-Disposition'] = 'attachment; filename=processed_images.zip'
        return response