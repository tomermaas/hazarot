"""
hazarot_subst package
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hazarot-subst")
except PackageNotFoundError:
    __version__ = "0.0.0"
