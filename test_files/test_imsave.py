from PIL import Image, ImageGrab
import numpy as np
import cv2
import os, sys
from datetime import datetime
import time
import concurrent.futures

time.sleep(10)

images = []
for i in range(1000):
    images.append(ImageGrab.grab())
    time.sleep(0.05)

input = input("Shall I start the test?")
print('\n\n\n')

def print_progress(pos, total, bins=50, before_msg=None, after_msg=None):
    bar = '[' + '#'*int((pos/total) * bins) + '-'*(bins - int((pos/total) * bins)) + ']'
    if not before_msg: before_msg = ''
    if not after_msg: after_msg = ''
    msg = f'\r{before_msg} {pos}/{total} {bar} {after_msg}  '
    if pos==total:
        print(msg)
    else:
        print(msg, end='  ')
        sys.stdout.flush()

def sim(image, filename):
    image.save(filename)

def sim_reduce(image, filename, size):
    image.resize(size).save(filename)

def sim_reduce_opencv(image, filename, size):
    image = cv2.resize(np.array(image), size)
    cv2.imwrite(filename, image)

def sim_opencv(image, filename):
    cv2.imwrite(filename, np.array(image))

def sim_opencv_rgb(image, filename):
    cv2.imwrite(filename, np.array(image)[:,:,::-1])

def test_imsave_function(images, save_function, args, folder, verbose=True):
    if not os.path.exists(os.path.join('test_files', 'test_save', folder)):
        os.makedirs(os.path.join('test_files', 'test_save', folder))
    t_start_total = datetime.now()
    for i, image in enumerate(images):
        if i%10==0 and verbose:
            t_start = datetime.now()
        save_function(image, os.path.join('test_files', 'test_save', folder, f'{i}.png'), *args)
        if i%10==0 and verbose:
            t_diff = datetime.now() - t_start
            t_est = datetime.now() + ((len(images) - i) * t_diff)
        if verbose:
            print_progress(i+1, len(images), bins=20, before_msg='Saving Images', after_msg=f'Est. finish time: {t_est.strftime("%I:%M %p")}')
    t_diff_total = datetime.now() - t_start_total
    if verbose:
        return t_diff_total

time = test_imsave_function(images, sim, [], 'sim_1')
print(f'\nTime taken to save {len(images)} images - {time}\n')

time = test_imsave_function(images, sim_opencv, [], 'sim_2')
print(f'\nTime taken to save {len(images)} images with opencv - {time}\n')

time = test_imsave_function(images, sim_opencv_rgb, [], 'sim_3')
print(f'\nTime taken to save {len(images)} images with opencv in RGB - {time}\n')

# time = test_imsave_function(images, sim_reduce, [(1280, 720)], 'sim_3')
# print(f'\nTime taken to save {len(images)} images by reduceing size to 1280x720 - {time}\n')
#
# time = test_imsave_function(images, sim_reduce_opencv, [(1280, 720)], 'sim_4')
# print(f'\nTime taken to save {len(images)} images by reduceing size to 1280x720 with opencv - {time}\n')
#
# time = test_imsave_function(images, sim_reduce, [(640, 480)], 'sim_5')
# print(f'\nTime taken to save {len(images)} images by reduceing size to 640x480 - {time}\n')
#
# time = test_imsave_function(images, sim_reduce_opencv, [(640, 480)], 'sim_6')
# print(f'\nTime taken to save {len(images)} images by reduceing size to 640x480 with opencv- {time}\n')


# def test_imsave_function_multithread(images, save_function, args, folder, workers):
#     if not os.path.exists(os.path.join('test_files', 'test_save', folder)):
#         os.makedirs(os.path.join('test_files', 'test_save', folder))
#     print('Started multithread saving...')
#     t_start_total = datetime.now()
#     image_chunks = [images[i:i + int(len(images)/workers)+1] for i in range(0, len(images), int(len(images)/workers)+1)]
#     arg_sections = [(chunk, save_function, args, folder, False) for chunk in image_chunks]
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for args in arg_sections:
#             futures.append(
#                 executor.submit(
#                     test_imsave_function, *args
#                 )
#             )
#         for future in concurrent.futures.as_completed(futures):
#             try: pass
#             except requests.ConnectTimeout:
#                 print("ConnectTimeout.")
#
#     t_diff_total = datetime.now() - t_start_total
#     return t_diff_total
#
# time = test_imsave_function_multithread(images, sim, [], 'sim_7', 4)
# print(f'\nTime taken to save {len(images)} images - {time}\n')
#
# time = test_imsave_function_multithread(images, sim_opencv, [], 'sim_8', 4)
# print(f'\nTime taken to save {len(images)} images with opencv - {time}\n')
#
# time = test_imsave_function_multithread(images, sim_reduce, [(1280, 720)], 'sim_9', 4)
# print(f'\nTime taken to save {len(images)} images by reduceing size to 1280x720 - {time}\n')
#
# time = test_imsave_function_multithread(images, sim_reduce_opencv, [(1280, 720)], 'sim_10', 4)
# print(f'\nTime taken to save {len(images)} images by reduceing size to 1280x720 with opencv - {time}\n')
#
# time = test_imsave_function_multithread(images, sim_reduce, [(640, 480)], 'sim_11', 4)
# print(f'\nTime taken to save {len(images)} images by reduceing size to 640x480 - {time}\n')
#
# time = test_imsave_function_multithread(images, sim_reduce_opencv, [(640, 480)], 'sim_12', 4)
# print(f'\nTime taken to save {len(images)} images by reduceing size to 640x480 with opencv- {time}\n')
