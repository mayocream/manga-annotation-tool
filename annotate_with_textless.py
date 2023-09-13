import os
import asyncio
from PIL import Image, ImageChops
import numpy as np
from skimage.morphology import opening, closing, disk
import logging
import argparse

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_images(img_path, textless_img_path, output_dir):
    async def load_image(path):
        return Image.open(path)

    im1 = await load_image(img_path)
    im2 = await load_image(textless_img_path)
    translucent = Image.new("RGBA", im1.size, (255, 0, 0, 127))
    mask = ImageChops.difference(im1, im2).convert("L").point(lambda x: 127 if x else 0)

    # Reduce noise on the mask
    mask_np = np.array(mask)
    # mask_np = reduce_noise(mask_np)
    mask = Image.fromarray(mask_np)

    im2.paste(translucent, (0, 0), mask)
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    im2.save(output_path)
    logger.info(f"Processed and saved {output_path}")

def reduce_noise(mask_np, selem_size=3):
    # Use opening to remove noise
    mask_np = opening(mask_np, disk(selem_size))
    # Use closing to fill gaps
    mask_np = closing(mask_np, disk(selem_size))
    return mask_np

async def main(directory_path, output_path="output"):
    img_dir = os.path.join(directory_path, 'original/')
    textless_img_dir = os.path.join(directory_path, 'textless/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    original_images = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')])
    textless_images = sorted([os.path.join(textless_img_dir, fname) for fname in os.listdir(textless_img_dir) if fname.endswith('.jpg')])

    logger.info(f"Found {len(original_images)} original images and {len(textless_images)} textless images.")

    tasks = []
    for img_path, textless_img_path in zip(original_images, textless_images):
        task = asyncio.create_task(process_images(img_path, textless_img_path, output_path))
        tasks.append(task)

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images from specified directory.")
    parser.add_argument("dir", help="Directory containing 'original' and 'textless' subdirectories.")
    parser.add_argument("out", nargs='?', default="output", help="Output directory. Defaults to 'output'.")
    args = parser.parse_args()

    asyncio.run(main(args.dir, args.out))
