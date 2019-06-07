import json


def read_json(f):
    """
    Read json file.

    Args:
      f: (str) file path
    """

    with open(f, 'r') as con:
        return json.load(con)


def write_json(x, f):
    """
    Save dict as json.

    Args:
      x: (dict) data
      f: (str) file path
    """

    with open(f, 'w') as con:
        json.dump(x, con, indent=4)


def print_json(x):
    """
    Print dict as (nicely formatted) json.

    Args:
      x: (dict) data
    """

    print(json.dumps(x, indent=4))


