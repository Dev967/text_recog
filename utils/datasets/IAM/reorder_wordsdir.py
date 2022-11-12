import os
import shutil

from CONF import *


def reorder():
    if host != "localhost":
        for subdir in os.listdir(image_dir):
            shutil.move(f'{image_dir}/{subdir}/*', f'{image_dir}')
            os.rmdir(subdir)
