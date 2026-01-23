#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Croptic Deep Learning Open documentation
#

import sys, os
sys.path.insert(0, os.path.abspath('../'))

# -- Project information
project = 'Croptic Deep Learning Open'
copyright = '2026, Croptic Team'
author = 'Croptic Team'

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

autodoc_member_order = 'alphabetical'

# -- HTML output
html_theme = 'alabaster'
html_theme_options = {
    'description': 'Open Source version of Croptic Deep Learning Library',
    'fixed_sidebar': True,
    'sidebar_width': '240px',
}

# TOC NAVIGATION (NumPy-style)
html_sidebars = {
    '**': [
        'localtoc.html',      # 📄 Current page TOC
        'relations.html',     # ← Next/Prev
        'searchbox.html',     # 🔍 Search
    ],
}

templates_path = ['_templates']
exclude_patterns = []

# Suppress noise
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
