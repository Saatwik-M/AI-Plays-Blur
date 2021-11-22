import tensorflow as tf
from helper_functions import key_check, grab_and_save_screen, split_image
import pydirectinput
import numpy as np
import cv2
from config import *
import sys
import time

model = tf.keras.models.load_model('models/test_baseline_model_4', compile=False)
# model = tf.keras.models.load_model('models/mathew_test_split_1')

for l in model.layers:
    l.trainable=False

# pred_keys = np.array(['w', 's', 'a', 'd', 'left', 'up', 'right', 'down'])
pred_keys = np.array(['w', 's', 'left', 'up', 'right', 'down'])

def preprocessImage(screen):
    # img = cv2.resize(cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY), (299, 299)).reshape((1, 299, 299, 1))/255
    img = cv2.resize(np.array(screen), (299, 299)).reshape((1, 299, 299, 3))/255

    # img_rv, img_fv, im_pv, im_mv = split_image(np.array(screen))
    # img_rv, img_fv, im_pv, im_mv = cv2.resize(img_rv, (101,35)).reshape((1, 101, 35, 3))/255, \
    #                                 cv2.resize(img_fv, (200,66)).reshape((1, 200, 66, 3))/255, \
    #                                 cv2.resize(cv2.cvtColor(im_pv, cv2.COLOR_BGR2GRAY), (53,20)).reshape((1, 53, 20, 1))/255, \
    #                                 cv2.resize(cv2.cvtColor(im_mv, cv2.COLOR_BGR2GRAY), (47,14)).reshape((1, 47, 14, 1))/255

    return img #img_rv, img_fv, im_pv, im_mv


keys = key_check(['c', 'e', 's', 'p', 'y', 'i'], VK_CODE, 'list')

start_playing = False
keys_to_press = []

while True:
    keys = key_check(['c', 'e', 's', 'p', 'y', 'i'], VK_CODE, 'list')
    if 'c' in keys and 'p' in keys:
        if not start_playing:
            start_playing = True
            print('\n')
            # Start collecting the data in 10 seconds
            for i in range(10):
                print(f'\rPlaying game in {9-i} seconds...', end='   ')
                sys.stdout.flush()
                time.sleep(1)
            print('\nStarted Playing')
            pydirectinput.keyDown('q')

    elif 's' in keys and 'p' in keys:
        start_playing = False
        pydirectinput.keyUp('q')
        print('\nStopped Playing')

    elif 'e' in keys and 'p' in keys:
        pydirectinput.keyUp('q')
        print('\nStopped the program')
        break

    if start_playing:
        pydirectinput.keyDown('q')
        _, screen = grab_and_save_screen()
        img = preprocessImage(screen)
        pred = model.predict(img)>0.5
        # img_rv, img_fv, im_pv, im_mv = preprocessImage(screen)
        # pred = model.predict([img_fv, img_rv, im_pv, im_mv])>0.5
        for key in keys_to_press:
            pydirectinput.keyUp(key)
        keys_to_press = pred_keys[pred[0]]
        print(keys_to_press)
        for key in keys_to_press:
            pydirectinput.keyDown(key)
