import platform

@property
def is_windows():
    return platform.system() == 'Windows'
