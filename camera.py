#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ RealSense Viewer (Fixed 4:3, Numeric Control Only)
 - RGB 영상 640x480 고정 비율
 - 창 크기 고정 (리사이징 불가)
 - 슬라이더 제거, 숫자 입력만
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
width, height = 640, 480
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)
pipeline.start(config)

device = pipeline.get_active_profile().get_device()
color_sensor = device.query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

# ============================================================
# ✅ Tkinter 창 설정 (리사이즈 불가, 고정)
# ============================================================
root = tk.Tk()
root.title("🎛️ RealSense RGB Control (Fixed 4:3)")
root.geometry("2400x1350")    # 고정된 넓은 창
root.resizable(False, False)  # 크기 변경 금지

# 프레임 구성
top_frame = ttk.Frame(root)
top_frame.pack(fill="both", expand=True)

bottom_frame = ttk.Frame(root)
bottom_frame.pack(fill="x", pady=10)

# ============================================================
# ✅ 영상 라벨 (640x480 비율 유지)
# ============================================================
canvas_width = 1600
canvas_height = int(canvas_width * 3 / 4)  # 4:3 비율

image_label = ttk.Label(top_frame)
image_label.place(relx=0.5, rely=0.5, anchor="center", width=canvas_width, height=canvas_height)

# ============================================================
# ✅ 제어 가능한 옵션 리스트
# ============================================================
OPTIONS = [
    ("Exposure", rs.option.exposure, 1, 1000),
    ("Gain", rs.option.gain, 0, 128),
    ("Brightness", rs.option.brightness, -64, 64),
    ("Contrast", rs.option.contrast, 0, 100),
    ("Gamma", rs.option.gamma, 100, 500),
    ("Hue", rs.option.hue, -180, 180),
    ("Saturation", rs.option.saturation, 0, 100),
    ("Sharpness", rs.option.sharpness, 0, 100),
    ("WhiteBalance", rs.option.white_balance, 2800, 6500),
]

entry_vars = {}

def make_entry(parent, name, option, minv, maxv):
    """숫자 입력 전용 옵션 컨트롤"""
    frame = ttk.Frame(parent)
    frame.pack(side="left", padx=15)

    ttk.Label(frame, text=name).pack()
    var = tk.DoubleVar()

    try:
        var.set(color_sensor.get_option(option))
    except Exception:
        var.set((minv + maxv) / 2)

    entry = ttk.Entry(frame, textvariable=var, width=10, justify="center")
    entry.pack()

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

# 입력칸 생성
for name, opt, mn, mx in OPTIONS:
    make_entry(bottom_frame, name, opt, mn, mx)

# Auto 옵션
auto_frame = ttk.Frame(bottom_frame)
auto_frame.pack(side="left", padx=20)
auto_wb = tk.BooleanVar(value=False)
auto_exp = tk.BooleanVar(value=False)

def toggle_auto_white():
    color_sensor.set_option(rs.option.enable_auto_white_balance, float(auto_wb.get()))

def toggle_auto_exposure():
    color_sensor.set_option(rs.option.enable_auto_exposure, float(auto_exp.get()))

ttk.Checkbutton(auto_frame, text="Auto WB", variable=auto_wb, command=toggle_auto_white).pack()
ttk.Checkbutton(auto_frame, text="Auto EXP", variable=auto_exp, command=toggle_auto_exposure).pack()

# ============================================================
# ✅ 영상 업데이트 루프 (비율 고정)
# ============================================================
def update_frame():
    global frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        root.after(10, update_frame)
        return

    frame = np.asanyarray(color_frame.get_data())
    resized = cv2.resize(frame, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)

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
    x = int(event.x * width / canvas_width)
    y = int(event.y * height / canvas_height)
    if 0 <= x < width and 0 <= y < height:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        h, s, v = hsv[y, x]
        l, a, b = lab[y, x]
        print(f"(x={x}, y={y}) → HSV=({h},{s},{v})  LAB=({l},{a},{b})")

image_label.bind("<Button-1>", on_click)

# ============================================================
# ✅ 실행 및 종료 처리
# ============================================================
root.after(100, update_frame)
root.protocol("WM_DELETE_WINDOW", lambda: (pipeline.stop(), root.destroy()))
root.mainloop()
