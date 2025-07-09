import os
import re

def safe_filename(name):
    name = re.sub(r'[^a-zA-Z0-9_\-\.]+', '_', name)
    return name 