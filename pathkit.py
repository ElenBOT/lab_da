"""Filepath kit using tkinter.

functions:
    get_filepath: Pop up a dialog to browse a file path (or save path) then return it.
    get_filepaths: Pop up a dialog to browse and select multiple files and returned as list.
    get_folderpath: Pop up a dialog to browse a folder path then return it.
    get_filepaths_under: Return file paths recursively from the given folder, can be filter by filename.
    replace_ext: Return a filepath with its file extension modified.
"""

__all__ = [
    'get_filepath',
    'get_filepaths',
    'get_folderpath',
    'get_filepaths_under',
    'replace_ext',
    'get_file_size',
]

from tkinter import Tk, filedialog
import ctypes
import os
from typing import Callable

def get_filepath(ext: str = None, title='Select a file path', initialdir: str = None, savefile=False):
    """Pop up a dialog to browse a file path (or save path) then return it.

    Example usage:
    >>> filepath = get_filepath('.hdf5', title='Measurement Data')
    
    Args:
        ext (str): The filename extension (e.g., '.txt', '.hdf5'). If None, all file types are allowed.
        title (str): The title displayed on the dialog.
        save_filepath (bool): If True, it asks for a path to save a file.

    Returns:
        filepath (str): The filepath, or an empty string if canceled.
    """
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    if savefile:
        dialog = filedialog.asksaveasfilename
    else:
        dialog = filedialog.askopenfilename

    if ext:
        # Ensure extension starts with a dot
        ext = ext if ext.startswith('.') else '.' + ext
        filetypes = [(ext.upper() + ' Files', '*' + ext)]
    else:
        filetypes = [('All Files', '*.*')]

    filepath = dialog(filetypes=filetypes, title=title, initialdir=initialdir)
    return filepath

def get_filepaths(ext: str = None, title='Select file(s)', initialdir: str = None):
    """Pop up a dialog to browse and select multiple files.

    Example usage:
    >>> filepaths = get_filepaths('.csv', title='Select CSV files')
    
    Args:
        ext (str): The file extension filter (e.g., '.csv'). If None, all files are allowed.
        title (str): The title displayed on the dialog.

    Returns:
        filepaths (list[str]): A list of selected file paths. Empty list if canceled.
    """
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    if ext:
        # Ensure extension starts with a dot
        ext = ext if ext.startswith('.') else '.' + ext
        filetypes = [(ext.upper() + ' Files', '*' + ext)]
    else:
        filetypes = [('All Files', '*.*')]

    filepaths = filedialog.askopenfilenames(filetypes=filetypes, title=title, initialdir=initialdir)
    return list(filepaths)

def get_folderpath(title='Select a folder', initialdir: str = None,):
    """Pop up a dialog to browse and select a folder.

    Example usage:
    >>> folder = get_folderpath('Select a data folder')
    
    Args:
        title (str): The title displayed on the dialog.

    Returns:
        str: The selected folder path. Empty string if canceled.
    """
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folderpath = filedialog.askdirectory(title=title, initialdir=initialdir)
    return folderpath

def get_filepaths_under(folderpath: str, filter: Callable[[str], bool] = None):
    """Return file paths recursively from the given folder, can be filter by filename.

    Example usage:
    >>> # No filtering
    >>> files = get_filepaths_under("C:/data")
    >>> # Get only CSV files
    >>> files = get_filepaths_under("C:/data", filter=lambda p: p.endswith('.csv'))
    >>> # Get files that contains `old_version`
    >>> files = get_filepaths_under("C:/data", filter=lambda p: 'old_version' in p)
    
    Args:
        folderpath (str): The path to the root folder.
        filter (Callable): A function that takes a file path and returns True to include it.

    Returns:
        list[str]: A list of full file paths found recursively that pass the filter.
    """
    all_files = []
    for root, _, files in os.walk(folderpath):
        for file in files:
            filepath = os.path.join(root, file)
            if filter is None or filter(filepath):
                all_files.append(filepath)
    return all_files

def replace_ext(filepath, new_ext: str):
    """Return a filepath with its file extension modified.

    Example usage:
    >>> new_path = modify_ext('audio.wav', '.mp3')
    """
    # Ensure the new extension starts with a dot
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    
    # Split the file path into root and extension
    root, _ = os.path.splitext(filepath)
    
    # Join the root with the new extension
    return os.path.join(os.path.dirname(filepath), root.split(os.sep)[-1] + new_ext)

def get_file_size(filepath):
    """Return the file size as a readable string like '12.56KB', '1.34MB', as well as number of bytes.
    (Generate by AI)

    Example usage:
    >>> get_readable_file_size("video.mkv")
    OUTPUT:
    | '284.88 MB', 298716659
    """
    size_bytes = os.path.getsize(filepath)
    if size_bytes == 0:
        return "0B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    value = size_bytes
    while value >= 1024 and i < len(units) - 1:
        value /= 1024.0
        i += 1
    return f"{value:.2f} {units[i]}", size_bytes