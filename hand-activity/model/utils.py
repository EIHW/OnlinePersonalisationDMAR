from os import makedirs
from os.path import exists

def make_directory(dir):
    if not exists(dir):
        makedirs(dir)