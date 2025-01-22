#Django Imports
from django.http import HttpResponse

from django.shortcuts import render, redirect
from django.urls import reverse
from django.conf import settings

import os
import numpy as np
import cv2
import torch
import shutil
import roifile
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torchvision.transforms as transforms

from heart_segmentation.unet.unet_model import UNet

"""Helper Function START"""

def save_uploaded_file(image):
    os.makedirs(settings.DATA_PATH, exist_ok=True)
    input_file_path = os.path.join(settings.DATA_PATH, "input.tif")

    with open(input_file_path, "wb") as file:
        for chunk in image.chunks():
            file.write(chunk)

def get_padded_image():
    image_file = os.path.join(settings.DATA_PATH, "input.tif")
    image_matrix = cv2.imread(image_file)
    original_shape = image_matrix.shape

    patch_size = settings.PATCH_SIZE
    total_height_padding = patch_size - (image_matrix.shape[0] % patch_size)
    total_width_padding = patch_size - (image_matrix.shape[1] % patch_size)
    height_padding = (total_height_padding // 2, total_height_padding - total_height_padding // 2)
    width_padding = (total_width_padding // 2, total_width_padding - total_width_padding // 2)
    padded_image = np.pad(image_matrix, (height_padding, width_padding, (0, 0)), mode="constant",
                    constant_values=255)

    return original_shape, padded_image 

def get_model(device):
    model = UNet(n_channels=3, n_classes=1, bilinear=True)

    model_weights = torch.load(settings.MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(model_weights)

    model.eval()

    model.to(device)

    return model

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def predict_mask(image):
    patch_size = settings.PATCH_SIZE
    number_of_patches = (image.shape[0] // patch_size) * (image.shape[1] // patch_size)

    pred_mask = np.zeros(image.shape[:2])

    height = (0, patch_size)
    width = (0, patch_size)

    device = get_device()
    model = get_model(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    for i in range(number_of_patches):
        curr_patch = image[height[0]:height[1], width[0]:width[1], :]

        black_pixels = np.all(curr_patch == [0, 0, 0], axis=-1)
        curr_patch[black_pixels] = [255, 255, 255]

        hls_image = cv2.cvtColor(curr_patch, cv2.COLOR_BGR2HLS)
        lightness_threshold = 150
        lightness = hls_image[:, :, 1]
        white_mask = (lightness >= lightness_threshold)
        is_all_white = np.all(white_mask)

        pred_patch = np.zeros(curr_patch.shape[:2])
        if not is_all_white:
            #Make a prediction
            input = transform(curr_patch).unsqueeze(0)
            input = input.to(device)

            with torch.no_grad():
                output = model(input)
            
            output = torch.round(torch.sigmoid(output.squeeze(1)))

            pred_patch = output.detach().cpu().numpy()
        
        pred_mask[height[0]:height[1], width[0]:width[1]] = pred_patch

        if width[1] == image.shape[1]:
            width = (0, patch_size)
            height = (height[0] + patch_size, height[1] + patch_size)
        else:
            width = (width[0] + patch_size, width[1] + patch_size)

    pred_mask = pred_mask * 255 
    return pred_mask

def save_mask(mask):
    mask = mask.astype(np.uint8)
    mask_path = os.path.join(settings.DATA_PATH, "mask.png")
    cv2.imwrite(mask_path, mask)

def reshape_pred_mask(original_shape, mask):
    height_excess = (mask.shape[0] - original_shape[0])
    width_excess = (mask.shape[1] - original_shape[1])

    height_bottom = height_excess // 2
    height_top = height_excess - height_bottom

    width_bottom = width_excess // 2
    width_top = width_excess - width_bottom

    return mask[height_bottom:mask.shape[0] - height_top, width_bottom:mask.shape[1] - width_top]

def make_rois_from_mask(mask):
    mask = mask.astype(np.uint8)
    roi_directory = os.path.join(settings.DATA_PATH, "rois")
    os.makedirs(roi_directory, exist_ok=True)

    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    curr_index = 1
    for _, contour in enumerate(contours):
        contour = contour.squeeze()
        contour = contour.astype(float)

        roi_output = roifile.ImagejRoi.frompoints(contour)

        output_file = os.path.join(roi_directory, f"{curr_index}.roi")
        roi_output.tofile(output_file)
        curr_index += 1
    
    folder_to_zip = roi_directory
    output_zip = os.path.join(settings.DATA_PATH, "rois")
    shutil.make_archive(output_zip, 'zip', folder_to_zip)

def predicted_mask_overlay():
    # Load the image 
    
    input_file_path = os.path.join(settings.DATA_PATH, "input.tif")
    
    result = cv2.imread(input_file_path)
    
    m1 = np.ones((result.shape[0], result.shape[1], 1))
    result = np.dstack((result, m1))
    result = result.astype(float)
    # m2 = np.zeros((result.shape[0], result.shape[1], 2)) 
    # mask_color = np.dstack((m1, m2))

    # result cv2.cvtColor(rgb_data, rgba , cv::COLOR_RGB2RGBA)

    # Load the ROI coordinates

    roi_dir = os.path.join(settings.DATA_PATH, "rois")

    # Store the results

    for file in os.listdir(roi_dir):

        f_path = os.path.join(roi_dir, file)

        roi = roifile.roiread(f_path)
        roi_coordinates_raw = roi.coordinates()
        roi_coordinates = []

        # Convert raw coordinates to tuples and add as separate contours
        contour = []
        for i, point in enumerate(roi_coordinates_raw):
            contour.append((point[0], point[1]))
            
            # If there's a significant jump between consecutive points, treat as new contour
            if i > 0 and np.linalg.norm(np.array(point) - np.array(roi_coordinates_raw[i - 1])) > 100:  # adjust 100 as needed
                roi_coordinates.append(contour)
                contour = []  # Start a new contour

        # Add last contour if not empty
        if contour:
            roi_coordinates.append(contour)

        # Create a mask image
        mask = Image.new("RGBA", (result.shape[1], result.shape[0]), (0,0,0,0))
        draw = ImageDraw.Draw(mask, 'RGBA')

        # Draw each contour individually to avoid connections between them
        for contour in roi_coordinates:
            draw.polygon(contour, outline=(1, 255, 1, 1), fill=(0, 0, 0, 0), width = 2)



        mask = np.array(mask)
        
        # mask = mask.astype(np.uint8)
        # Blend the mask with the image

        print(mask.dtype)
        print(result.dtype)
        # result = cv2.addWeighted(result, 1, mask, 0.5, 0.4)
        result = np.where(mask > 0, mask, result)
        


        # Display and save the result
    result = result[:,:,:3]
    os.makedirs(settings.DATA_PATH, exist_ok=True)
    output = os.path.join(settings.DATA_PATH, "overlay.png")
    cv2.imwrite(output, result)
       


"""Helper Function END"""

# Create your views here.
def index(request):
    if os.path.exists(settings.DATA_PATH) and os.path.isdir(settings.DATA_PATH):
        shutil.rmtree(settings.DATA_PATH)
    return render(request, "index.html", {})

def process_image(request):
    image = request.FILES.get("image", None)
    save_uploaded_file(image)
    original_shape, padded_image = get_padded_image()
    mask = predict_mask(padded_image)
    mask = reshape_pred_mask(original_shape, mask)
    save_mask(mask)
    make_rois_from_mask(mask)
    predicted_mask_overlay()
    return redirect(reverse("heart_segmentation:result"))

def result(request):
    roi_zip_dir = os.path.join(settings.DATA_PATH, "rois.zip")
    input_file_path = os.path.join(settings.DATA_PATH, "overlay.png")

    roi_zip_dir = f"{settings.MEDIA_URL}rois.zip"
    input_file_path = f"{settings.MEDIA_URL}overlay.png"

    context = {
        "roi_zip_dir": roi_zip_dir,
        "input_file_path": input_file_path
    }
    return render(request, "result.html", context)