import os


def validate_correct_img_path(path):

    return path and os.path.exists(path) and os.path.getsize(path) > 0
