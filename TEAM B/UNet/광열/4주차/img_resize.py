import os
import cv2
from tqdm import tqdm

def resize_images(input_folder, output_folder, size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = os.listdir(input_folder)

    for img_file in tqdm(image_files):
        input_path = os.path.join(input_folder, img_file)
        
        image = cv2.imread(input_path)
        
        resized_image = cv2.resize(image, (size[1], size[0]))
        
        output_path = os.path.join(output_folder, img_file)
        
        cv2.imwrite(output_path, resized_image)

if __name__ == "__main__":

    input_folder = "./data/masks"
    output_folder = "./data2/masks"
    new_size = (512, 512)

    resize_images(input_folder, output_folder, new_size)
