from recommonmark.transform import AutoStructify

project = "modes"
version = "1.0.2"
copyright = "2019, Jean-Luc Tambasco"
author = "Jean-Luc Tambasco"

master_doc = "index"
html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}


extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_markdown_tables",
    "sphinx.ext.doctest",
    "recommonmark",
    "sphinx_autodoc_typehints",
]

autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "build",
    "extra/**",
]

napoleon_use_param = True


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {"auto_toc_tree_section": "Contents", "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
