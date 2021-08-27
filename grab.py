# Importing Necessary Modules
import requests  # to get image from the web
import shutil  # to save it locally
import time
import os

from time import strftime

# Set up the image URL and other constants
image_url = "https://fc.sseinfo.com/Captcha"
total = 200
interval = 0.5  # sec
folder = "raw_images"
img_format = "png"


def download(url, dest):
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(url, stream=True)
    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        # Open a local file with wb ( write binary ) permission.
        with open(dest, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print('Image sucessfully Downloaded: ', dest)
    else:
        print('Image Couldn\'t be retreived. URL: %s', url)


if __name__ == '__main__':
    os.makedirs(folder, exist_ok=True)
    for i in range(total):
        img_path = os.path.join(folder, "img_{}.{}".format(i, img_format))
        download(image_url, img_path)
        time.sleep(interval)
