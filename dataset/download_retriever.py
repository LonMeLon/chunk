import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


os.system('huggingface-cli download --resume-download --local-dir-use-symlinks False BAAI/bge-large-en-v1.5 --local-dir bge-large-en-v1.5')