# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../src'))

# import gazetimation
# from gazetimation import Gazetimation

project = 'Gazetimation'
copyright = '2022, Shuvo Paul'
author = 'Shuvo Paul'
release = '0.0.1'

html_logo = "assets/gazetimation_logo.png"
html_favicon = "assets/favicon.ico"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = []
autoclass_content = "both"



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/paul-shuvo/gazetimation/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
html_static_path = ['_static']
