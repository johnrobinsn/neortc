# Put your OpenAI key here or
# create .neortc.json in your home directory with key: openai_api_key

api_key=''

if not api_key:
    import os
    import json
    from pathlib import Path

    home_directory = Path(os.path.expanduser('~'))
    filename = home_directory/'.neortc.json'
    with open(filename, 'r') as file:
        data = json.load(file)
        api_key = data.get('openai_api_key','')
