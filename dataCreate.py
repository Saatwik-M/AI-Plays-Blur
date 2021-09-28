import pandas as pd, win32api as wapi, time, os, sys
from datetime import datetime
from PIL import ImageGrab
from helper_functions import key_check, grab_and_save_screen, save_data
from config import *
import gc


print("----- Started Program (press 'cp' to start, 'sp' to stop and 'ep' to end) -----\n")

# Initialising parameters
all_keys = []
all_screens = {}
collect_data = False
saved_data = True
file_name = ''

# Main program loop
try:
    while True:
        # Keys needed to start the program
        keys = key_check(['c', 'e', 's', 'p'], VK_CODE, 'list')

        # 'c' and 'p' to start collecting the data
        if 'c' in keys and 'p' in keys:
            if not collect_data:
                collect_data = True
                saved_data = False
                print('\n')
                # Start collecting the data in 10 seconds
                for i in range(10):
                    print(f'\rData collection starting in {9-i} seconds...', end='   ')
                    sys.stdout.flush()
                    time.sleep(1)
                print('\nStarted Collecting Data')
                screen_time = round(time.time() * 1000)

        # 's' and 'p' to pause collecting the data
        elif 's' in keys and 'p' in keys:
            # If already collecting the data...
            if not saved_data:
                # Calculate the amount of data collected till now and average time gap between screenshots
                time_stamps = [int(k.split('.')[0]) for k in all_screens]
                time_gaps = []
                for i in range(len(time_stamps)-1):
                    time_gaps.append(time_stamps[i+1]-time_stamps[i])
                avg_time = int(sum(time_gaps) / len(time_gaps))
                print(f'Stopping Collecting Data (Data length - {len(time_stamps)}, Avg timegap - {avg_time}ms)')
                collect_data = False
                saved_data = True
            # If not collecting the data
            else:
                print("\nProgram not running, press 'cp' to start")
            time.sleep(1)
            gc.collect() # Remove garbage values to save memory

        # 'e' and 'p' to end the program
        elif 'e' in keys and 'p' in keys:
            # If program is not paused...
            if not saved_data:
                time_stamps = [int(k.split('.')[0]) for k in all_screens]
                time_gaps = []
                for i in range(len(time_stamps)-1):
                    time_gaps.append(time_stamps[i+1]-time_stamps[i])
                avg_time = int(sum(time_gaps) / len(time_gaps))
                print(f'Stopping Collecting Data (Data length - {len(time_stamps)}, Avg time gap - {avg_time}ms)')
            # Save the screenshots and the keys strokes
            print('\n')
            save_data(all_screens, all_keys, raw_data_folder)
            print('\n\n----- Ending the program -----')
            break

        # If the program is collecting data...
        if collect_data:
            # Try to collect screenshots only every 60 milli seconds
            if round(time.time() * 1000) >= screen_time + 50:
                screen_time = round(time.time() * 1000)
                file_name, screenshot = grab_and_save_screen(raw_data_folder, False)
                all_screens[file_name] = screenshot
            # Get the keypresses along with timestamps
            keys = key_check(keys_to_check, VK_CODE, 'dict')
            all_keys.append(keys)
            print(f'>>> file - {file_name}, keys - {[key_short_forms[k] for k,v in keys.items() if v != 0 and k != "time"]}', end=' \r')
            sys.stdout.write("\033[K") # Clear output line
            time.sleep(0.01)

# Save data even in case of keyboard inturrupt
except KeyboardInterrupt:
    if not saved_data:
        time_stamps = [int(k.split('.')[0]) for k in all_screens]
        time_gaps = []
        for i in range(len(time_stamps)-1):
            time_gaps.append(time_stamps[i+1]-time_stamps[i])
        avg_time = int(sum(time_gaps) / len(time_gaps))
        print(f'Stopping Collecting Data (Data length - {len(time_stamps)}, Avg time gap - {avg_time}ms)')
    print('\n')
    save_data(all_screens, all_keys, raw_data_folder)
    print('\n\n----- Ending the program -----')
