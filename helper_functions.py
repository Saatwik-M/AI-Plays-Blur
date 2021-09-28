import pandas as pd, win32api as wapi, time, os, sys
from datetime import datetime
from PIL import ImageGrab
from config import *


def key_check(keys_to_check, key_codes, return_type):
    """
    Function to capture key presses return as a list or a dictionary.

    Parameters
    ----------
        keys_to_check : seq
            the keys to check if they are pressed
        key_codes : dict
            key letter to keycode mappings
        return_type : str
            'list' or 'dict'

    Returns
    -------
        list of keys presses or dict of all keys with values 1 if pressed
    """
    if return_type == 'list':
        keys = []
    else:
        keys = dict.fromkeys(keys_to_check, 0)
        keys['time'] = round(time.time() * 1000)
    for key in keys_to_check:
        if wapi.GetAsyncKeyState(key_codes[key]):
            if return_type == 'list': keys.append(key)
            else: keys[key] = 1
    return keys


def grab_and_save_screen(folder=None, save=False):
    """
    Function to capture screenshot and save if specified.

    Parameters
    ----------
        folder : str
            folder to save
        save : bool
            To save or not

    Returns
    -------
        filename if saved or filename plus screenshot if not saved
    """
    file_name = str(round(time.time() * 1000)) + '.png'
    screenshot = ImageGrab.grab()
    if save:
        filepath = os.path.join(folder, 'screenshots', file_name)
        screenshot.save(filepath, 'PNG')
        return file_name
    else:
        return file_name, screenshot


def print_progress(pos, total, bins=50, before_msg=None, after_msg=None):
    """
    Prints progress bar with before and after messages.

    Parameters
    ----------
        pos : int
            folder to save
        total : int
            To save or not
        bins : int
            folder to save
        before_msg : str
            To save or not
        after_msg : str
            folder to save

    Returns
    -------
        None
    """
    bar = '[' + '#'*int((pos/total) * bins) + '-'*(bins - int((pos/total) * bins)) + ']'
    if not before_msg: before_msg = ''
    if not after_msg: after_msg = ''
    msg = f'\r{before_msg} {pos}/{total} {bar} {after_msg}  '
    if pos==total:
        print(msg)
    else:
        print(msg, end='  ')
        sys.stdout.flush()

def save_data(screens, key_strokes, folder):
    """
    Save data .

    Parameters
    ----------
        screens : str
            screenshots to save
        key_strokes : bool
            key strokes to save
        folder : str
            folder to save

    Returns
    -------
        None
    """
    data = pd.DataFrame(key_strokes).sort_values('time', ascending=True).reset_index(drop=True)
    if os.path.isfile(os.path.join(folder, 'keyStrokesRaw.csv')):
        old_data = pd.read_csv(os.path.join(folder, 'keyStrokesRaw.csv'))
        data = pd.concat([data, old_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
    data.to_csv(os.path.join(folder, 'keyStrokesRaw.csv'), index=False)
    print('Saved Keystrokes')
    for i, file_name in enumerate(screens):
        print_progress(i+1, len(screens), bins=50, before_msg='Saving Images')
        filepath = os.path.join(folder, 'screenshots', file_name)
        screens[file_name].save(filepath, 'PNG')
