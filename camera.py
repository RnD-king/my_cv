#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ RealSense Viewer (Fixed 840x900, Mask Toggle + HSV/LAB Range)
 - 4:3 비율 유지 (640x480)
 - Auto WB / EXP 토글
 - Space로 Pause
 - HSV/LAB 기준 색상 마스크
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
# ✅ Tkinter 창 설정
# ============================================================
root = tk.Tk()
root.title("🎛️ RealSense RGB Control (Mask Pro Edition)")
root.geometry("840x900")
root.resizable(False, False)
root.attributes('-fullscreen', False)

# ============================================================
# ✅ 프레임 구성
# ============================================================
frame_video = ttk.Frame(root, width=840, height=630)
frame_video.pack_propagate(False)
frame_video.pack()

frame_params = ttk.Frame(root, height=90)
frame_params.pack(fill="x", pady=3)

frame_mask = ttk.LabelFrame(root, text="🎨 Mask Settings", padding=5)
frame_mask.pack(fill="x", pady=3)

frame_status = ttk.Frame(root, height=30)
frame_status.pack(fill="x", pady=(0, 5))

# ============================================================
# ✅ 영상 라벨
# ============================================================
image_label = ttk.Label(frame_video)
image_label.place(relx=0.5, rely=0.5, anchor="center")

# ============================================================
# ✅ 일반 설정 입력창 (노출, 밝기 등)
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
# ✅ Auto 옵션 (WB / EXP)
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
# ✅ Mask 설정 (HSV/LAB 선택 + 범위 입력)
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

# HSV
h_vars = make_range_inputs(frame_mask, "H")
s_vars = make_range_inputs(frame_mask, "S")
v_vars = make_range_inputs(frame_mask, "V")
# LAB
l_vars = make_range_inputs(frame_mask, "L")
a_vars = make_range_inputs(frame_mask, "A")
b_vars = make_range_inputs(frame_mask, "B")

# 모드 선택 + 마스크 토글
mode_frame = ttk.Frame(frame_mask)
mode_frame.pack(side="left", padx=15)
ttk.Checkbutton(mode_frame, text="Mask ON", variable=mask_on).pack(anchor="w")
ttk.Radiobutton(mode_frame, text="HSV", variable=mask_mode, value="HSV").pack(anchor="w")
ttk.Radiobutton(mode_frame, text="LAB", variable=mask_mode, value="LAB").pack(anchor="w")

# ============================================================
# ✅ 상태 표시 + Space 일시정지
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
# ✅ 영상 업데이트 루프
# ============================================================
frame = None

def apply_mask(img):
    """현재 설정값 기준으로 마스크 적용"""
    if not mask_on.get():
        return img
    mode = mask_mode.get()
    if mode == "HSV":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([h_vars[0].get(), s_vars[0].get(), v_vars[0].get()], dtype=np.uint8)
        upper = np.array([h_vars[1].get(), s_vars[1].get(), v_vars[1].get()], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        lower = np.array([l_vars[0].get(), a_vars[0].get(), b_vars[0].get()], dtype=np.uint8)
        upper = np.array([l_vars[1].get(), a_vars[1].get(), b_vars[1].get()], dtype=np.uint8)
        mask = cv2.inRange(lab, lower, upper)

    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked

def update_frame():
    global frame
    if not paused:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame = np.asanyarray(color_frame.get_data())

    if frame is not None:
        img = apply_mask(frame.copy())
        # 4:3 비율 유지
        target_w, target_h = 840, 630
        aspect_src = cam_w / cam_h
        aspect_dst = target_w / target_h
        if aspect_dst > aspect_src:
            new_h = target_h
            new_w = int(new_h * aspect_src)
        else:
            new_w = target_w
            new_h = int(new_w / aspect_src)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
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
