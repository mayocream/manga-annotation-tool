import os
import cv2
import asyncio
import numpy as np
from skimage.morphology import opening, closing, disk
from PIL import Image, ImageChops, ImageDraw
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
    mask = ImageChops.difference(im1, im2).convert("L").point(lambda x: 255 if x > 0 else 0)  # Binary mask

    # Reduce noise on the mask
    mask_np = np.array(mask)
    mask_np = reduce_noise(mask_np)
    mask = Image.fromarray(mask_np)

    # Convert PIL mask to OpenCV format
    mask_cv = np.array(mask)

    # Find contours
    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    polygons = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x+w, y+h))

        # Get polygon and reshape
        polygon = contour.reshape(-1, 2)
        polygon_list = [(point[0], point[1]) for point in polygon]

        if len(polygon_list) >= 2:  # Ensure we have at least 2 coordinates
            polygons.append(polygon_list)

    # Preview
    im_preview = im1.copy()
    draw = ImageDraw.Draw(im_preview)
    for box in bounding_boxes:
        draw.rectangle(box, outline=(255, 0, 0))
    for polygon in polygons:
        draw.polygon(polygon, outline=(0, 255, 0))

    preview_path = os.path.join(output_dir, f"preview_{os.path.basename(img_path)}")
    im_preview.save(preview_path)
    logger.info(f"Saved preview {preview_path}")

    # Further saving or processing of bounding_boxes and polygons can be added

    output_path = os.path.join(output_dir, os.path.basename(img_path))
    if mask is not None:
        im2.paste(translucent, (0, 0), mask)
    else:
        im2.paste(translucent, (0, 0))

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
