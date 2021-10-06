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
    """
    Apply random augmentations to the image
    img: ndarray
    """
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    combo = iaa.SomeOf((0,5),
                   [iaa.Add((-15, 15), per_channel=0.5),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    sometimes(iaa.ElasticTransformation(alpha=(0, img.shape[0]//5), sigma=img.shape[0]//50)),
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),
                    iaa.Grayscale(alpha=(0.0, 0.3)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.Affine(shear=(0,10)),
                    iaa.OneOf([iaa.GaussianBlur((0, 3)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                               iaa.MotionBlur(k=(5,10))]),
                    iaa.Multiply((0.8, 1.2), per_channel=0.5)], random_order=True)
    return (combo.augment_image(img))

def get_key_pressed(im_time, time_window, key_pd, criteria='max'):
    """
    Gets the keys pressed for a particular image for a specific period of time

    parameters
    ----------
    im_time: timestamp of the image
    time_window: time interval for getting key strocks
    key_pd: DataFrame of keys csv file
    criteria: 'max' --> to take maximum values of all columns of key dataframe for the time interval
              else --> it takes majority value of all columns of the key dataframe for the time interval

    Returns
    -------
    list
    """
    keys = key_pd[(key_pd['time']>= im_time) & (key_pd['time'] <= im_time + time_window)]
    if criteria =='max':
        return (list(keys.max())[:-1])
    else:
        mode = keys.mode(axis=0, numeric_only=True, dropna=True)
        return list(map(lambda x: int(x), mode.iloc[0].values))[:-1]

def combine_keypresses(folder, time_window, criteria='max'):
    """
    It combines key presses for all images and store it in a csv file
    Parameters
    ----------
    time_window: time interval for getting key strocks
    criteria: 'max' --> to take maximum values of all columns of key dataframe for the time interval
              else --> it takes majority value of all columns of the key dataframe for the time interval
    folder: the main directory
    """
    key_list = []
    img_list = sorted(list(map(lambda x: int(x.split('.')[0]), os.listdir(os.path.join(folder,"screenshots")))))
    key_pd = pd.read_csv(os.path.join(folder, 'keyStrokesRaw.csv'))
    for time in img_list:
        key_list.append(get_key_pressed(time, time_window, key_pd, criteria) + [time])
    cmd_keypd = pd.DataFrame(key_list, columns=key_pd.columns)
    cmd_keypd.to_csv(os.path.join(folder,"combinedKeyStrokes.csv"))

def image_resize(folder, normal=True, split=False, aug=False):
    """
    The function resizes the image and/or split the image with/without augmentations based on the options
    Parameters
    ----------
    normal: True --> resizes the original image
    split: True --> splits the image into 4 parts and resizes
    aug: True --> apply various augmentations to the image
    It stores the updated image in separate direcory

    Returns
    -------
    None
    """
    resize_dict = {'normal':(299, 299), 'im_fv':(200,66), 'im_rv':(100,24), 'im_pv':(100,24), 'im_mv':(48,48)}
    if normal:
        if not os.path.isdir(os.path.join(folder, 'resized_img')):
            os.mkdir(os.path.join(folder, 'resized_img'))
    if split:
        if not os.path.isdir(os.path.join(folder, 'split_img')):
            os.mkdir(os.path.join(folder, 'split_img'))
    screenshot_dir = os.path.join(folder,"screenshots")
    for files in os.listdir(screenshot_dir):
        im = np.array(Image.open(os.path.join(screenshot_dir,files)))
        if normal:
            im_n = np.array(Image.fromarray(im).resize(resize_dict['normal']))
            if aug:
                im_n = aug_combo(im_n)
            Image.fromarray(im_n).save(os.path.join(folder, 'resized_img',files))

        if split:
            im_rv, im_fv, im_pv, im_mv  = split_img(im)
            im_rv, im_fv, im_pv, im_mv = (np.array(Image.fromarray(im_rv).resize(resize_dict['im_rv'])),
                                          np.array(Image.fromarray(im_fv).resize(resize_dict['im_fv'])),
                                          np.array(Image.fromarray(im_pv).resize(resize_dict['im_fv'])),
                                          np.array(Image.fromarray(im_mv).resize(resize_dict['im_fv'])))
            if aug:
                im_rv, im_fv, im_pv, im_mv = (aug_combo(im_rv),aug_combo(im_fv),aug_combo(im_pv),aug_combo(im_rv))
            Image.fromarray(im_rv).save(os.path.join(folder, 'split_img', '%s_rear.png' % files.split('.')[0]))
            Image.fromarray(im_fv).save(os.path.join(folder, 'split_img', '%s_front.png' % files.split('.')[0]))
            Image.fromarray(im_pv).save(os.path.join(folder, 'split_img', '%s_power.png' % files.split('.')[0]))
            Image.fromarray(im_mv).save(os.path.join(folder, 'split_img', '%s_map.png' % files.split('.')[0]))
