import platform

def add_sys_path_to_dll_load_path():
    import os
    for path in os.environ["PATH"].split(";"):
        try:
            os.add_dll_directory(path)
        except:
            pass

# After Python 3.8, environment variable PATH and the current working
# directory are no longer used for searching dll's load-time dependencies.
# We need to add these path manually, especially when we want to link cudart.
# See:
# - https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew
# - https://docs.python.org/3/library/os.html#os.add_dll_directory
if platform.system() == "Windows":
    py_version = platform.python_version_tuple()
    py_major_version, py_minor_version = int(py_version[0]), int(py_version[1])
    if py_major_version >= 3 and py_minor_version >= 8:
        add_sys_path_to_dll_load_path()

from . import pytensorlite_C as _C
