#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ RealSense Viewer (Fixed 840x900, Mask with Circular Hue Support)
 - HSV/LAB 마스크 지원
 - H축 원형 (min>max → wraparound)
 - Auto WB / EXP 토글, Pause, HSV/LAB 클릭 출력
"""

import tkinter as tk
from tkinter import ttk
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image, ImageTk

# ============================================================
# ✅ RealSense 초기화
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
cam_w, cam_h = 640, 480
config.enable_stream(rs.stream.color, cam_w, cam_h, rs.format.bgr8, 15)
pipeline.start(config)

device = pipeline.get_active_profile().get_device()
color_sensor = device.query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

# ============================================================
# ✅ Tkinter GUI 기본 세팅
# ============================================================
root = tk.Tk()
root.title("🎛️ RealSense RGB Control (Mask Hue Circular)")
root.geometry("840x900")
root.resizable(False, False)
root.attributes('-fullscreen', False)

frame_video = ttk.Frame(root, width=840, height=630)
frame_video.pack_propagate(False)
frame_video.pack()

frame_params = ttk.Frame(root, height=90)
frame_params.pack(fill="x", pady=3)

frame_mask = ttk.LabelFrame(root, text="🎨 Mask Settings", padding=5)
frame_mask.pack(fill="x", pady=3)

frame_status = ttk.Frame(root, height=30)
frame_status.pack(fill="x", pady=(0, 5))

image_label = ttk.Label(frame_video)
image_label.place(relx=0.5, rely=0.5, anchor="center")

# ============================================================
# ✅ 기본 파라미터 입력
# ============================================================
OPTIONS = [
    ("Exposure", rs.option.exposure, 1, 1000),
    ("Gain", rs.option.gain, 0, 128),
    ("Brightness", rs.option.brightness, -64, 64),
    ("Contrast", rs.option.contrast, 0, 100),
    ("Gamma", rs.option.gamma, 100, 500),
    ("Saturation", rs.option.saturation, 0, 100),
    ("Sharpness", rs.option.sharpness, 0, 100),
    ("WhiteBalance", rs.option.white_balance, 2800, 6500),
]

entry_vars = {}

def make_entry(parent, name, option, minv, maxv):
    frame = ttk.Frame(parent)
    frame.pack(side="left", padx=6)
    ttk.Label(frame, text=name, font=("Arial", 9)).pack()
    var = tk.DoubleVar()
    try:
        var.set(color_sensor.get_option(option))
    except Exception:
        var.set((minv + maxv) / 2)
    entry = ttk.Entry(frame, textvariable=var, width=8, justify="center")
    entry.pack(pady=2)
    def apply_value(event=None):
        try:
            value = float(var.get())
            if value < minv: value = minv
            if value > maxv: value = maxv
            color_sensor.set_option(option, value)
            var.set(value)
        except ValueError:
            pass
    entry.bind("<Return>", apply_value)
    entry_vars[name] = var

for name, opt, mn, mx in OPTIONS:
    make_entry(frame_params, name, opt, mn, mx)

# ============================================================
# ✅ Auto 옵션
# ============================================================
auto_frame = ttk.Frame(frame_params)
auto_frame.pack(side="left", padx=10)
auto_wb = tk.BooleanVar(value=False)
auto_exp = tk.BooleanVar(value=False)
def toggle_auto_white():
    color_sensor.set_option(rs.option.enable_auto_white_balance, float(auto_wb.get()))
def toggle_auto_exposure():
    color_sensor.set_option(rs.option.enable_auto_exposure, float(auto_exp.get()))
ttk.Checkbutton(auto_frame, text="Auto WB", variable=auto_wb, command=toggle_auto_white).pack()
ttk.Checkbutton(auto_frame, text="Auto EXP", variable=auto_exp, command=toggle_auto_exposure).pack()

# ============================================================
# ✅ Mask UI (HSV/LAB + 범위 입력)
# ============================================================
mask_on = tk.BooleanVar(value=False)
mask_mode = tk.StringVar(value="HSV")

def make_range_inputs(parent, name):
    frm = ttk.LabelFrame(parent, text=name, padding=3)
    frm.pack(side="left", padx=5)
    vars_minmax = []
    for label in ["min", "max"]:
        row = ttk.Frame(frm)
        row.pack()
        ttk.Label(row, text=f"{label}:", width=5).pack(side="left")
        var = tk.DoubleVar(value=0 if label == "min" else 255)
        ent = ttk.Entry(row, textvariable=var, width=6, justify="center")
        ent.pack(side="left", padx=2)
        vars_minmax.append(var)
    return vars_minmax

h_vars = make_range_inputs(frame_mask, "H")
s_vars = make_range_inputs(frame_mask, "S")
v_vars = make_range_inputs(frame_mask, "V")
l_vars = make_range_inputs(frame_mask, "L")
a_vars = make_range_inputs(frame_mask, "A")
b_vars = make_range_inputs(frame_mask, "B")

mode_frame = ttk.Frame(frame_mask)
mode_frame.pack(side="left", padx=15)
ttk.Checkbutton(mode_frame, text="Mask ON", variable=mask_on).pack(anchor="w")
ttk.Radiobutton(mode_frame, text="HSV", variable=mask_mode, value="HSV").pack(anchor="w")
ttk.Radiobutton(mode_frame, text="LAB", variable=mask_mode, value="LAB").pack(anchor="w")

# ============================================================
# ✅ 상태 라벨 + Spacebar 일시정지
# ============================================================
status_label = ttk.Label(frame_status, text="Status: Playing", font=("Arial", 11, "bold"))
status_label.pack(anchor="center")

paused = False
def toggle_pause(event=None):
    global paused
    paused = not paused
    status_label.config(text=f"Status: {'Paused' if paused else 'Playing'}")

root.bind("<space>", toggle_pause)

# ============================================================
# ✅ 마스크 적용 함수 (Hue 원형 대응)
# ============================================================
def apply_mask(img):
    if not mask_on.get():
        return img

    if mask_mode.get() == "HSV":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min, h_max = h_vars[0].get(), h_vars[1].get()
        s_min, s_max = s_vars[0].get(), s_vars[1].get()
        v_min, v_max = v_vars[0].get(), v_vars[1].get()

        # H는 원형(0~180)
        if h_min <= h_max:
            mask_h = cv2.inRange(hsv[:,:,0], h_min, h_max)
        else:
            # 예: 170~180, 0~10
            mask_h1 = cv2.inRange(hsv[:,:,0], h_min, 180)
            mask_h2 = cv2.inRange(hsv[:,:,0], 0, h_max)
            mask_h = cv2.bitwise_or(mask_h1, mask_h2)

        mask_s = cv2.inRange(hsv[:,:,1], s_min, s_max)
        mask_v = cv2.inRange(hsv[:,:,2], v_min, v_max)
        mask = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_s, mask_v))
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l_min, l_max = l_vars[0].get(), l_vars[1].get()
        a_min, a_max = a_vars[0].get(), a_vars[1].get()
        b_min, b_max = b_vars[0].get(), b_vars[1].get()
        mask_l = cv2.inRange(lab[:,:,0], l_min, l_max)
        mask_a = cv2.inRange(lab[:,:,1], a_min, a_max)
        mask_b = cv2.inRange(lab[:,:,2], b_min, b_max)
        mask = cv2.bitwise_and(mask_l, cv2.bitwise_and(mask_a, mask_b))

    return cv2.bitwise_and(img, img, mask=mask)

# ============================================================
# ✅ 영상 업데이트 루프
# ============================================================
frame = None
def update_frame():
    global frame
    if not paused:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame = np.asanyarray(color_frame.get_data())
    if frame is not None:
        img = apply_mask(frame.copy())
        target_w, target_h = 840, 630
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=image)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)
    root.after(20, update_frame)

# ============================================================
# ✅ 클릭 시 HSV/LAB 값 출력
# ============================================================
def on_click(event):
    if frame is None:
        return
    x = int(event.x * cam_w / 840)
    y = int(event.y * cam_h / 630)
    if 0 <= x < cam_w and 0 <= y < cam_h:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        h, s, v = hsv[y, x]
        l, a, b = lab[y, x]
        print(f"(x={x}, y={y}) HSV=({h},{s},{v})  LAB=({l},{a},{b})")

image_label.bind("<Button-1>", on_click)

# ============================================================
# ✅ 실행
# ============================================================
root.after(100, update_frame)
root.protocol("WM_DELETE_WINDOW", lambda: (pipeline.stop(), root.destroy()))
root.mainloop()
