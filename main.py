import torch
import cv2
import mss
import pyautogui
import time
import pygetwindow as gw
import numpy as np

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'bestv2.pt',
                       force_reload=True, trust_repo=True)

def capture_roblox_screen(roblox_window):
    if roblox_window is None:
        return None
    
  
    left, top, width, height = roblox_window.left, roblox_window.top, roblox_window.width, roblox_window.height
    
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

def detect_objects(image):
    results = model(image)
    return results.xyxy[0]

def move_cursor_to_box(box, roblox_window):
    if roblox_window is None:
        return
    
    box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    relative_x = roblox_window.left + box_center[0]
    relative_y = roblox_window.top + box_center[1]
    
    pyautogui.moveTo(relative_x, relative_y)
    pyautogui.click()

def get_roblox_window():
    windows = gw.getWindowsWithTitle('Roblox')
    if windows:
        return windows[0] 
    return None


while True:
    roblox_window = get_roblox_window()
    
    if roblox_window is None:
        print("Roblox window not found, stopping cursor control.")
        time.sleep(1)  
        continue
    
    frame = capture_roblox_screen(roblox_window)
    if frame is not None:
        results = detect_objects(frame)
        boxes = results[:, :4]  
        
        if len(boxes) > 0:
            
            for box in boxes:
                move_cursor_to_box(box, roblox_window)
                time.sleep(0.5)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

