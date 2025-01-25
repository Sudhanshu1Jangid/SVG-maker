import cv2
import numpy as np
import ctypes
import os
import subprocess
import ctypes
import os

# Set the path to the directory containing the cairo DLL
cairo_dll_path = r'C:\GTK\bin\libcairo-2.dll'
import ctypes

# Path to the correct Cairo DLL (make sure this is the 64-bit version)
cairo_dll_path = r'C:\msys64\mingw64\bin\libcairo-2.dll'  # Update the path accordingly

try:
    ctypes.windll.LoadLibrary(cairo_dll_path)
    print("Cairo DLL loaded successfully!")
except Exception as e:
    print(f"Error loading Cairo DLL: {e}")


# Now import cairosvg
from cairosvg import svg2png

def capture_image():
    cap = cv2.VideoCapture('http://----------/video') #enter your ip camera address
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press SPACE to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

import ctypes

def get_scale(image):
    # Get the screen resolution in Windows using ctypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    width_screen = user32.GetSystemMetrics(0)
    height_screen = user32.GetSystemMetrics(1)

    # Calculate the scale based on the image dimensions and screen resolution
    height_image, width_image = image.shape[:2]
    image_ar = width_image / height_image
    
    if width_image <= height_image:
        return height_screen * 0.7 / height_image
    else:
        return width_screen * 0.4 / width_image

        
    
def select_roi(image):

    scale_factor = get_scale(image)

    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (width, height))

    roi_scaled = cv2.selectROI("Select area to crop", resized_image)
    roi = tuple(map(lambda x: int(x/scale_factor) , roi_scaled))
    cv2.destroyAllWindows()
    return roi

def crop_image(image, roi):
    return image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

def save_image(image, filename):
    cv2.imwrite(filename, image)

def convert_to_pbm(input_file, output_file, threshold):
    subprocess.run(['magick', input_file, '-threshold', f'{threshold}%', output_file], shell=True)


def convert_to_svg(input_file, output_file):
    subprocess.run(['potrace', '-s', '--tight', '-o', output_file, input_file])  # No -threshold argument


def convert_to_png(input_file, output_file):
    with open(input_file, 'r') as svg_file:
        svg_code = svg_file.read()
        svg2png(bytestring=svg_code,write_to=output_file)


def main():
    # Create results folder if it doesn't exist
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Capture image
    image = capture_image()

    
    # Select ROI and crop
    roi = select_roi(image)
    cropped_image = crop_image(image, roi)
    
    # Generate unique filename
    keep = False
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    while not keep:
        threshold = float(input('Threshold: '))
        temp_name = 'temp'
        
        # Save cropped image
        jpg_filename = os.path.join(results_folder, f"{temp_name}.jpg")
        save_image(cropped_ixmage, jpg_filename)

        # Convert to PBM
        pbm_filename = os.path.join(results_folder, f"{temp_name}.pbm")
        convert_to_pbm(jpg_filename, pbm_filename, threshold)
        
        # Convert to SVG
        svg_filename = os.path.join(results_folder, f"{temp_name}.svg")
        convert_to_svg(pbm_filename, svg_filename)
        
        # Convert to PNG
        png_filename = os.path.join(results_folder, f"{temp_name}.png")
        convert_to_png(svg_filename, png_filename)

        preview = cv2.imread(png_filename, cv2.IMREAD_UNCHANGED)
        mask = preview[:, :, 3]
        scale_factor = get_scale(mask)
        height_mask = int(mask.shape[0] * scale_factor)
        width_mask = int(mask.shape[1] * scale_factor)
        mask = cv2.resize(mask, (height_mask, width_mask) )
        #preview = cv2.resize(preview, (int(preview.shape[1]*0.2), int(preview.shape[0]*0.2)))
        
        #cv2.imshow('Preview', preview)
        cv2.imshow('mask', mask)
        k = cv2.waitKey(0)

        if k == 13:
            cv2.destroyAllWindows()
            keep = not keep
            base_filename = input("What diagram is this? \n")
        cv2.destroyAllWindows()
            
    os.remove(jpg_filename)
    os.remove(pbm_filename)
    os.remove(svg_filename)
    new_path = os.path.join(os.path.dirname(png_filename), f'{base_filename}.png')
    os.rename(png_filename, new_path)
    
    
    print(f"PNG file saved as: {base_filename}")

if __name__ == "__main__":
    main()