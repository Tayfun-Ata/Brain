from pynput import keyboard, mouse
import json
import threading
import time

# File to store collected data
DATA_FILE = "user_interactions.json"

# Shared data structure
data = []
stop_flag = False  # Flag to stop the script

def save_data_periodically():
    while not stop_flag:
        time.sleep(10)  # Save every 10 seconds
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print("Data saved.")

def on_key_press(key):
    global stop_flag
    try:
        if key == keyboard.Key.esc:  # Stop the script when Esc is pressed
            stop_flag = True
            print("Stopping data collection...")
            return False  # Stop the keyboard listener
        data.append({
            "type": "key_press",
            "key": key.char,
            "timestamp": time.time()
        })
    except AttributeError:
        data.append({
            "type": "key_press",
            "key": str(key),
            "timestamp": time.time()
        })

def on_click(x, y, button, pressed):
    if stop_flag:
        return False  # Stop the mouse listener
    data.append({
        "type": "mouse_click",
        "button": str(button),
        "pressed": pressed,
        "position": (x, y),
        "timestamp": time.time()
    })

def on_move(x, y):
    if stop_flag:
        return False  # Stop the mouse listener
    data.append({
        "type": "mouse_move",
        "position": (x, y),
        "timestamp": time.time()
    })

def start_data_collection():
    global stop_flag
    # Start saving data periodically in a separate thread
    save_thread = threading.Thread(target=save_data_periodically, daemon=True)
    save_thread.start()

    # Start keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener.start()

    # Start mouse listener
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    mouse_listener.start()

    print("Data collection started.")

    # Wait for the listeners to stop
    try:
        keyboard_listener.join()
        mouse_listener.stop()  # Stop the mouse listener when the keyboard listener stops
        mouse_listener.join()
    except KeyboardInterrupt:
        stop_flag = True
        print("Data collection interrupted.")
    finally:
        stop_flag = True
        print("Data collection stopped.")
