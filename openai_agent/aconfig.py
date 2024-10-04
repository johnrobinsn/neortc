# Put your OpenAI key here or
# create .neortc.json in your home directory with key: openai_api_key
import os
import json
from pathlib import Path
from collections import ChainMap

defaults = {
    'neortc_port':8444,
    'neortc_token':'',
    'openai_api_key':'',
}

home_directory = Path(os.path.expanduser('~'))
filename = home_directory/'.neortc.json'
config_file = {}
try:
    with open(filename, 'r') as file:
        config_file = json.load(file)
except FileNotFoundError:
    print(f"WARNING: Configure neortc by createa file '{filename}'")

config = ChainMap(config_file, defaults)



