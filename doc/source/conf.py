import sys
import os
import time
import doctest

sys.path.insert(0, os.path.abspath("../../src"))

library_name = "jaxtransform3d"
project = library_name
copyright = u"2025-{}, Alexander Fabisch".format(time.strftime("%Y"))
author = "Alexander Fabisch"
release = __import__(library_name).__version__

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
    "sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
exclude_patterns = []

# theme
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
}
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

# autodoc
autodoc_default_options = {"member-order": "bysource"}
autosummary_generate = True  # generate files at doc/source/_apidoc
class_members_toctree = False
numpydoc_show_class_members = False

# doctest
doctest_default_flags = doctest.ELLIPSIS

# intersphinx
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
intersphinx_timeout = 10

# sphinx gallery
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "_auto_examples",
    "reference_url": {library_name: None},
    "filename_pattern": "/plot_",
    "image_scrapers": ("matplotlib"),
    "backreferences_dir": "_auto_examples/backreferences",
    "doc_module": library_name,
}
