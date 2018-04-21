import os

from utils.platform import is_windows

def extended_path(path):
    assert is_windows
    path = os.path.abspath(path)
    if path.startswith(r'\\'):
        return r'\\?\UNC' + '\\' + path[2:]
    return r'\\?' + '\\' + path
