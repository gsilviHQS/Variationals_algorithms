# -*- coding: utf-8 -*-

import sys, os

sys.path.insert(0, os.path.abspath('extensions'))

extensions = []

todo_include_todos = True
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = []
add_function_parentheses = True
#add_module_names = True
# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

project = u'NEASQC Project'
copyright = u'2022, Copyright'

version = '1.0'
release = '1.0'

# -- Options for HTML output ---------------------------------------------------

html_theme = 'sphinx_rtd_theme'
#html_theme_path = ['sphinx_rtd_theme']
html_title = "default title"
#html_short_title = None
#html_logo = None
#html_favicon = None
html_static_path = ['_static']
html_domain_indices = False
html_use_index = False
html_show_sphinx = False
htmlhelp_basename = 'help_basename'
html_show_sourcelink = False

def setup(app):
     from sphinx.util.texescape import tex_replacements
     tex_replacements += [(u'♮', u'$\\natural$'),
                          (u'ē', u'\=e'),
                          (u'♩', u'\quarternote'),
                          (u'↑', u'$\\uparrow$'),
                          ]
