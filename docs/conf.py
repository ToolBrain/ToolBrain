# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ToolBrain'
copyright = '2025, ToolBrain Team'
author = 'ToolBrain Team'

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from toolbrain import __version__

version = __version__
release = __version__
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
         'sphinx.ext.autosummary',
        'sphinx.ext.napoleon', 
        'myst_parser',
        'nbsphinx',
    ]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
