# BarcodeDetection
- Python project capable of detecting and outlining position of barcodes on images and videos, entirely implemented through mathematical image-processing operations.
- Detection algorithm activated by executing CS373_barcode_detection.py with the only argument being the filename of the image in `/images` folder, result is written to the 'output_images' folder.
- CS373_extension.py is the same algorithm used for mp4 videos instead of images, except the input and output folders are now `/videos` and `/output_videos` respectively.
- The algorithm works best for 700x700 images with one barcode and clear lighting, but it can still be improved for better detection.
- Some of the image processing techniques used are:

## Pipeline
![image](https://github.com/calebWei/BarcodeDetection/assets/100410646/f5257e8b-06fe-4606-a29a-d39e0f1fa921)

## Successful Examples of the Detected Barcode (Indicated by Green Box)
### 1
![image](https://github.com/calebWei/BarcodeDetection/assets/100410646/c42b9714-a8f7-4c9a-aa59-3aef78e60f13)
### 2
![image](https://github.com/calebWei/BarcodeDetection/assets/100410646/5065ccd0-7e01-48be-a215-7d8d1d349fe2)
### 3
![image](https://github.com/calebWei/BarcodeDetection/assets/100410646/5b96c818-98b0-4ce9-8cbe-8bb26980aee7)
