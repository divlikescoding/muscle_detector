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
    for i, contour in enumerate(contours):
        contour = contour.squeeze()
        contour = contour.astype(float)

        roi_output = roifile.ImagejRoi.frompoints(contour)

        output_file = os.path.join(roi_directory, f"{curr_index}.roi")
        roi_output.tofile(output_file)
        curr_index += 1
    
    folder_to_zip = roi_directory
    output_zip = os.path.join(settings.DATA_PATH, "rois")
    shutil.make_archive(output_zip, 'zip', folder_to_zip)


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
    return redirect(reverse("heart_segmentation:result"))

def result(request):
    roi_zip_dir = os.path.join(settings.DATA_PATH, "rois.zip")

    roi_zip_dir = f"{settings.MEDIA_URL}rois.zip"
    context = {
        "roi_zip_dir": roi_zip_dir
    }
    return render(request, "result.html", context)