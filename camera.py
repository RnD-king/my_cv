#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ RealSense Viewer (Video 840x630 + Parameter Panel Below)
 - 영상: 840x630 고정 크기
 - 파라미터: 영상 하단에 숫자 입력칸
 - 창 전체 크기: 840x780 (리사이즈 불가)
 - 클릭 시 HSV/LAB 출력
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
# ✅ Tkinter 설정 (창 고정)
# ============================================================
root = tk.Tk()
root.title("🎛️ RealSense RGB Control (Video 840x630)")
root.geometry("840x780")        # 영상(630) + 입력패널(150)
root.resizable(False, False)
root.attributes('-fullscreen', False)

# ============================================================
# ✅ 프레임 구성
# ============================================================
frame_video = ttk.Frame(root, width=840, height=630)
frame_video.pack_propagate(False)  # 내부 위젯 크기에 영향 안 받게
frame_video.pack()

frame_bottom = ttk.Frame(root, height=150)
frame_bottom.pack(fill="x", pady=5)

# ============================================================
# ✅ 영상 표시용 라벨 (840x630 고정)
# ============================================================
display_w, display_h = 840, 630
image_label = ttk.Label(frame_video)
image_label.place(x=0, y=0, width=display_w, height=display_h)

# ============================================================
# ✅ 제어 가능한 옵션 리스트
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
    """숫자 입력 전용 컨트롤"""
    frame = ttk.Frame(parent)
    frame.pack(side="left", padx=8)

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

# 숫자 입력 필드 생성
for name, opt, mn, mx in OPTIONS:
    make_entry(frame_bottom, name, opt, mn, mx)

# Auto 옵션
auto_frame = ttk.Frame(frame_bottom)
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
# ✅ 영상 업데이트 루프 (840x630에 맞게 리사이즈)
# ============================================================
def update_frame():
    global frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        root.after(10, update_frame)
        return

    frame = np.asanyarray(color_frame.get_data())
    resized = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

    image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=image)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

    root.after(20, update_frame)

# ============================================================
# ✅ 클릭 시 HSV/LAB 출력
# ============================================================
def on_click(event):
    if frame is None:
        return
    x = int(event.x * cam_w / display_w)
    y = int(event.y * cam_h / display_h)
    if 0 <= x < cam_w and 0 <= y < cam_h:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        h, s, v = hsv[y, x]
        l, a, b = lab[y, x]
        print(f"(x={x}, y={y}) → HSV=({h},{s},{v})  LAB=({l},{a},{b})")

image_label.bind("<Button-1>", on_click)

# ============================================================
# ✅ 종료 처리
# ============================================================
root.after(100, update_frame)
root.protocol("WM_DELETE_WINDOW", lambda: (pipeline.stop(), root.destroy()))
root.mainloop()
