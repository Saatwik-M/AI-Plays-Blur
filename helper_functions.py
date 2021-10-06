import pandas as pd, numpy as np
import cv2, win32api as wapi
from PIL import ImageGrab
from imgaug import augmenters as iaa
from datetime import datetime
import time, os, sys
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
        if i%10==0:
            t_start = datetime.now()
        filepath = os.path.join(folder, 'screenshots', file_name)
        # screens[file_name].save(filepath, 'PNG')
        cv2.imwrite(filepath, np.array(screens[file_name])[:,:,::-1])
        if i%10==0:
            t_diff = datetime.now() - t_start
            t_est = datetime.now() + ((len(screens) - i) * t_diff)
        print_progress(i+1, len(screens), bins=20, before_msg='Saving Images: ', after_msg=f'  Est. finish time: {t_est.strftime("%I:%M %p")}')

def split_image(image):
    """
    Splits images to 4 images .

    Parameters
    ----------
        image : ndarray
            image to split
        spectrum : str
            screenshots to save

    Returns
    -------
        None
    """
    rear = image[80:215, 530:1397]
    power = image[850:1015, 650:1250]
    front = image[480:586, 500:1450]
    maps = image[80:260, 1600:1700]
    return (rear, front, power, maps)

def aug_combo(img):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    combo = iaa.SomeOf((0,9),
                   [iaa.Add((-10, 10), per_channel=0.5),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.Affine(shear=(0,15)),
                    iaa.Invert(0.3, per_channel=True),
                    iaa.OneOf([iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                               iaa.MotionBlur(k=5)]),

                    iaa.Multiply((0.5, 1.5), per_channel=0.5)], random_order=True)
    return (combo.augment_image(img))

def get_key_pressed(im_time, time_window, key_pd, criteria='max'):
    keys = key_pd[(key_pd['time']>= im_time) & (key_pd['time'] <= im_time + time_window)]
    if criteria =='max':
        return (list(keys.max())[:-1])
    else:
        mode = keys.mode(axis=0, numeric_only=True, dropna=True)
        return list(map(lambda x: int(x), mode.iloc[0].values))[:-1]
