# Nothing serious here... Toy auth scheme
# Configure an auth "secret/token"
# Do it here locally -or-
# create .neortc.json in your home directory with key: neortc_secret

neortc_secret=''

if not neortc_secret:
    import os
    import json
    from pathlib import Path

    home_directory = Path(os.path.expanduser('~'))
    filename = home_directory/'.neortc.json'
    with open(filename, 'r') as file:
        data = json.load(file)
        neortc_secret = data.get('neortc_secret','')

if not neortc_secret:
    print('WARNING: Authentication disabled.  No Secret Configured')