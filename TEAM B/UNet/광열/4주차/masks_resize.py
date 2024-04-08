from PIL import Image
import os
from tqdm import tqdm

def resize_image(input_path, output_path, size):
    # 이미지 열기
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.BILINEAR)
        resized_img.save(output_path)

if __name__ == "__main__":

    input_folder = "./data/masks"
    output_folder = "./data2/masks"
    desired_size = (512, 512)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".gif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, desired_size)
