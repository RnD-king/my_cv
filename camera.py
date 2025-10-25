#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ RealSense Viewer (Fixed 840x780, Auto WB Once Lock)
 - 4:3 영상 비율 유지 (640x480)
 - Auto WB: 실시간 표시 + 1회 보정 후 고정 버튼 추가
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
# ✅ Tkinter 창 설정
# ============================================================
root = tk.Tk()
root.title("🎛️ RealSense RGB Control (Auto WB Once Lock)")
root.geometry("840x780")
root.resizable(False, False)
root.attributes('-fullscreen', False)

# ============================================================
# ✅ 프레임 구성
# ============================================================
frame_video = ttk.Frame(root, width=840, height=630)
frame_video.pack_propagate(False)
frame_video.pack()

frame_params = ttk.Frame(root, height=90)
frame_params.pack(fill="x", pady=5)

frame_wb_status = ttk.Frame(root, height=30)
frame_wb_status.pack(fill="x", pady=(0, 5))

# ============================================================
# ✅ 영상 라벨
# ============================================================
image_label = ttk.Label(frame_video)
image_label.place(relx=0.5, rely=0.5, anchor="center")

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
# ✅ Auto 옵션 + WB 표시 + 1회 보정 버튼
# ============================================================
auto_frame = ttk.Frame(frame_params)
auto_frame.pack(side="left", padx=10)

auto_wb = tk.BooleanVar(value=False)
auto_exp = tk.BooleanVar(value=False)

def toggle_auto_white():
    color_sensor.set_option(rs.option.enable_auto_white_balance, float(auto_wb.get()))
    if auto_wb.get():
        update_auto_wb_label()
    else:
        wb_label.config(text="Auto WB: OFF")

def toggle_auto_exposure():
    color_sensor.set_option(rs.option.enable_auto_exposure, float(auto_exp.get()))

# --- Auto WB 실시간 표시 ---
wb_label = ttk.Label(frame_wb_status, text="Auto WB: OFF", font=("Arial", 11, "bold"), anchor="center")
wb_label.pack(anchor="center")

def update_auto_wb_label():
    """Auto WB 값(K)을 주기적으로 읽어와 표시"""
    if auto_wb.get():
        try:
            current_wb = color_sensor.get_option(rs.option.white_balance)
            wb_label.config(text=f"Auto WB: {current_wb:.1f} K")
        except Exception:
            wb_label.config(text="Auto WB: N/A")
        root.after(500, update_auto_wb_label)

# --- Auto WB Once 기능 ---
def apply_auto_wb_once():
    """자동 WB를 1회 수행 후 그 값으로 고정"""
    wb_label.config(text="Auto WB Adjusting...")
    color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
    root.after(1500, lock_auto_wb_value)  # 1.5초 후 자동WB값 읽기 및 고정

def lock_auto_wb_value():
    try:
        current_wb = color_sensor.get_option(rs.option.white_balance)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, current_wb)
        wb_label.config(text=f"Locked WB: {current_wb:.1f} K")
        print(f"Auto-adjusted WB locked at {current_wb:.1f} K")
    except Exception as e:
        wb_label.config(text="WB Lock Error")
        print("WB Lock Error:", e)

# --- 버튼/체크박스 UI ---
ttk.Checkbutton(auto_frame, text="Auto WB", variable=auto_wb, command=toggle_auto_white).pack()
ttk.Checkbutton(auto_frame, text="Auto EXP", variable=auto_exp, command=toggle_auto_exposure).pack()

once_btn = ttk.Button(frame_wb_status, text="Auto Adjust Once", command=apply_auto_wb_once)
once_btn.pack(side="right", padx=15)

# ============================================================
# ✅ 영상 업데이트 루프 (4:3 비율 유지)
# ============================================================
def update_frame():
    global frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        root.after(10, update_frame)
        return

    frame = np.asanyarray(color_frame.get_data())

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

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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
# ✅ 클릭 시 HSV/LAB 출력
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
        print(f"(x={x}, y={y}) → HSV=({h},{s},{v})  LAB=({l},{a},{b})")

image_label.bind("<Button-1>", on_click)

# ============================================================
# ✅ 실행 및 종료
# ============================================================
root.after(100, update_frame)
root.protocol("WM_DELETE_WINDOW", lambda: (pipeline.stop(), root.destroy()))
root.mainloop()
