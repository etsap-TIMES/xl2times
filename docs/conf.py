# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "xl2times"
copyright = "2024, xl2times authors"
author = "xl2times authors"
# TODO version = xl2times.__version__ ?

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.apidoc",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_old"]

apidoc_module_dir = "../xl2times"
apidoc_output_dir = "api"
# apidoc_excluded_paths = ['tests']
# apidoc_separate_modules = True

# To allow `Attributes` block in class docstrings. TODO can we docstring the dataclass attributes directly?
napoleon_use_ivar = True

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/etsap-TIMES/xl2times",
    "use_repository_button": True,
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "home_page_in_toc": True,
}
