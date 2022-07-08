import sys
import os
import shlex
import subprocess

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

project = 'TensorRay'
authors = 'Edward Liu, Lifan Wu, Kai Yan, Cheng Zhang, Shuang Zhao'

version = '0.1.0'
release = '0.1.0'

language = None

exclude_patterns = ['.build', 'release.rst']

default_role = 'any'

todo_include_todos = False

import guzzle_sphinx_theme

html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = 'guzzle_sphinx_theme'
html_static_path = ['_static']

extensions = []
extensions.append("guzzle_sphinx_theme")
extensions.append("sphinx.ext.mathjax")

html_theme_options = {
    "project_nav_name": "TensorRay"
}

html_sidebars = {
    '**': ['logo-text.html', 'globaltoc.html', 'searchbox.html']
}

def setup(app):
    app.add_css_file('theme_overrides.css')

html_logo = None

html_static_path = ['_static']

html_show_sourcelink = False

htmlhelp_basename = 'enokidoc'

latex_elements = {
  'preamble': '\\DeclareUnicodeCharacter{00A0}{}',
}

latex_documents = [
  (master_doc, 'tensorRay.tex', 'TensorRay Documentation', authors, 'manual'),
]

man_pages = [
  (master_doc, 'tensorRay', 'TensorRay Documentation', authors, 1)
]

texinfo_documents = [
  (master_doc, 'tensorRay', 'TensorRay Documentation', authors, 'tensorRay', 'One line description of TensorRay.', 'Miscellaneous'),
]

primary_domain = 'cpp'
highlight_language = 'cpp'
