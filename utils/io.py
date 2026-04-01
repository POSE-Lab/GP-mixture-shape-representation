import yaml
import os
from rich.console import Console
from contextlib import contextmanager
import pandas as pd
import common
import sys

CONSOLE = Console()


def make_dir(path: str) -> None:
    """
    Creates a directory if it does not exist.
    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_config(path: str):
    """ Loads a configuration file from a path.

    Args:
        path (str): Path to configuration file.

    Returns:
        dict: Configuration file.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def log_event(message):
        CONSOLE.log(message)

def log_debug(message):
    if common.DEBUG:
        CONSOLE.log(message)

def loadEstimatorResults(path):

    return pd.read_csv(path)

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd) 
        sys.stdout = os.fdopen(fd, "w") 

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) 