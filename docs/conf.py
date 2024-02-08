# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "xl2times"
copyright = "2024, xl2times authors"
author = "xl2times authors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinxcontrib.apidoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # "sphinx_immaterial",
    # "sphinx_immaterial.apidoc.python.apigen",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_old"]

apidoc_module_dir = "../xl2times"
apidoc_output_dir = "api"
# apidoc_excluded_paths = ['tests']
# apidoc_separate_modules = True

# This didn't work: didn't create pages for classes, just table summary..
# # Autosummary
# autosummary_generate = True

# This didn't work because it tried to gen docs for DataFrame..
# python_apigen_modules = {
#     "xl2times.datatypes": "api/xl2times.datatypes.",
#     "xl2times.transforms": "api/xl2times.transforms.",
# }
# python_apigen_default_groups = [
#     (r".*:xl2times.datatypes.*", "datatypes-members"),
#     (r"class:xl2times.datatypes(?!DataFrame$).*", "datatypes-classes"),
#     (r".*:xl2times.transforms.*", "transforms-members"),
#     (r"class:xl2times.transforms.*", "transforms-classes"),
#     (r".*:.*\.__(init|new)__", "Constructors"),
#     (r".*:.*\.__eq__", "Comparison operators"),
#     (r".*:.*\.__(str|repr)__", "String representation"),
# ]

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
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/etsap-TIMES/xl2times",
    "use_repository_button": True,
    "path_to_docs": "docs",
    "use_edit_page_button": True,
}
