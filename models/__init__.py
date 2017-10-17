<<<<<<< HEAD
__all__ = []

import pkgutil
import inspect

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[name] = value
        __all__.append(name)
=======
from .mcnn import *
from .pnet import *
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
