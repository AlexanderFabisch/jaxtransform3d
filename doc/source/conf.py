import sys
import os
import time

sys.path.insert(0, os.path.abspath("../../src"))

project = "jaxtransform3d"
copyright = u"2025-{}, Alexander Fabisch".format(time.strftime("%Y"))
author = "Alexander Fabisch"
release = __import__("jaxtransform3d").__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    #"sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
}
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
intersphinx_timeout = 10
