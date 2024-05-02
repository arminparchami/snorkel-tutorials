import gdown
import os

url = 'https://drive.google.com/uc?id=1OkIMhApPHrzekSntIPKVkRQGGiYHFxHk'
output = 'sg_dataset.zip'
gdown.download(url, output, quiet=False)