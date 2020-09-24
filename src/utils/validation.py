import os

__all__ = [
    'is_dir', 'is_hdf'
]

def yn_input(text):
    choice = input(text).lower()
    while True:
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            choice = input(text + "enter [y/n] ").lower()


def is_dir(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return path
    else:
        raise None

def is_hdf(path):
    path = os.path.abspath(path)
    assert os.path.splitext(path)[-1] in ('.h5', '.hdf5')
    if os.path.exists(path):
        return path
    else:
        raise None

def new_hdf(path):
    path = os.path.abspath(path)
    assert os.path.splitext(path)[-1] in ('.h5', '.hdf5')
    if os.path.exists(path):
        if yn_input(f"{path} is already exists, overwrite? "):
            return path
        else:
            print("exit...")
            exit()
    else:
        return path
