# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# mock deps with system level requirements.
autodoc_mock_imports = ["soundfile"]

# -- Project information -----------------------------------------------------
project = 'TTS'
copyright = "2021 Coqui GmbH, 2020 TTS authors"
author = 'Coqui GmbH'

with open("../../TTS/VERSION", "r") as ver:
    version = ver.read().strip()

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
release = version

# The main toctree document.
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
    "sphinx_copybutton",
    "sphinx_inline_tabs",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'TODO/*']

source_suffix = [".rst", ".md"]

myst_enable_extensions = ['linkify',]

# 'sphinxcontrib.katex',
# 'sphinx.ext.autosectionlabel',


# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

language = 'en'

autodoc_inherit_docstrings = False

# Disable displaying type annotations, these can be very verbose
autodoc_typehints = 'none'

# Enable overriding of function signatures in the first line of the docstring.
autodoc_docstring_signature = True

napoleon_custom_sections = [('Shapes', 'shape')]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_tite = "TTS"
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
    "sidebar_hide_name": True,
}

html_sidebars = {
        '**': [
               "sidebar/scroll-start.html",
    "sidebar/brand.html",
    "sidebar/search.html",
    "sidebar/navigation.html",
    "sidebar/ethical-ads.html",
    "sidebar/scroll-end.html",
        ]
    }


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
