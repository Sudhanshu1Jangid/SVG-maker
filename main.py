import cv2
import numpy as np
import ctypes
import os
import subprocess
from datetime import datetime
from cairosvg import svg2png

def capture_image():
    cap = cv2.VideoCapture('http://your_camera_ip/video')  # Replace with your camera URL
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        cv2.imshow('Press SPACE to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

def get_scale(image):
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    width_screen = user32.GetSystemMetrics(0)
    height_screen = user32.GetSystemMetrics(1)

    height_image, width_image = image.shape[:2]
    if height_image == 0 or width_image == 0:
        raise ValueError("Invalid image dimensions.")

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
    if roi_scaled[2] <= 0 or roi_scaled[3] <= 0:
        print("Invalid ROI selected.")
        return None

    roi = tuple(map(lambda x: int(x / scale_factor), roi_scaled))
    cv2.destroyAllWindows()
    return roi

def crop_image(image, roi):
    return image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

def save_image(image, filename):
    cv2.imwrite(filename, image)

def convert_to_pbm(input_file, output_file, threshold):
    result = subprocess.run(['magick', input_file, '-threshold', f'{threshold}%', output_file], shell=True)
    if result.returncode != 0:
        raise RuntimeError("ImageMagick command failed.")

def convert_to_svg(input_file, output_file):
    result = subprocess.run(['potrace', '-s', '--tight', '-o', output_file, input_file], shell=True)
    if result.returncode != 0:
        raise RuntimeError("Potrace command failed.")

def convert_to_png(input_file, output_file):
    with open(input_file, 'r') as svg_file:
        svg_code = svg_file.read()
        svg2png(bytestring=svg_code, write_to=output_file)

def main():
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    image = capture_image()
    if image is None:
        return

    roi = select_roi(image)
    if roi is None:
        return

    cropped_image = crop_image(image, roi)

    while True:
        try:
            threshold = float(input('Enter threshold (0-100): '))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        temp_name = 'temp'

        jpg_filename = os.path.join(results_folder, f"{temp_name}.jpg")
        save_image(cropped_image, jpg_filename)

        pbm_filename = os.path.join(results_folder, f"{temp_name}.pbm")
        convert_to_pbm(jpg_filename, pbm_filename, threshold)

        svg_filename = os.path.join(results_folder, f"{temp_name}.svg")
        convert_to_svg(pbm_filename, svg_filename)

        png_filename = os.path.join(results_folder, f"{temp_name}.png")
        convert_to_png(svg_filename, png_filename)

        try:
            preview = cv2.imread(png_filename, cv2.IMREAD_UNCHANGED)
            if preview is None or preview.shape[2] < 4:
                raise ValueError("Invalid PNG file or missing alpha channel.")

            mask = preview[:, :, 3]
            scale_factor = get_scale(mask)
            height_mask = int(mask.shape[0] * scale_factor)
            width_mask = int(mask.shape[1] * scale_factor)
            mask = cv2.resize(mask, (width_mask, height_mask))

            cv2.imshow('Preview', mask)
            k = cv2.waitKey(0)

            if k == 13:  # Enter key
                cv2.destroyAllWindows()
                base_filename = input("Enter diagram name: ")
                break
        except Exception as e:
            print(f"Error previewing image: {e}")
        finally:
            cv2.destroyAllWindows()

    os.remove(jpg_filename)
    os.remove(pbm_filename)
    os.remove(svg_filename)

    new_path = os.path.join(results_folder, f'{base_filename}.png')
    os.rename(png_filename, new_path)
    print(f"PNG file saved as: {base_filename}.png")

if __name__ == "__main__":
    main()
