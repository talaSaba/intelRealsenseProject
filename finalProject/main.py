import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import os
import time
import torch
import tkinter as tk
from tkinter import ttk
from threading import Thread
from collections import OrderedDict
from YoloDetector import YoloDetector
from deep_sort_realtime.deepsort_tracker import DeepSort
from person import person_in_pool

# Global variables
watersurface = 0.5  # default value
our_Array = OrderedDict()  # array to maintain the people underwater
people_above = OrderedDict()  # array to maintain people above water

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Drowning Detection System")

    # Set window size
    root.geometry("800x600")

    # Styling options
    style = ttk.Style()
    style.theme_use('clam')  # 'clam', 'alt', 'default', 'classic'
    style.configure("TButton", background="#4CAF50", foreground="white", font=('Arial', 12), padding=10)
    style.configure("TLabel", background="#f0f0f0", font=('Arial', 12))
    style.configure("TEntry", padding=10)
    style.configure("Treeview.Heading", font=('Arial', 12, 'bold'))
    style.configure("Treeview", font=('Arial', 12), rowheight=30)

    # Frame for user input
    input_frame = tk.Frame(root, bg="#f0f0f0", bd=2, relief=tk.RIDGE)
    input_frame.pack(pady=20, padx=20, fill=tk.X)

    label = ttk.Label(input_frame, text="Enter Water Surface Distance:")
    label.pack(side=tk.LEFT, padx=5)

    entry = ttk.Entry(input_frame, font=('Arial', 12))
    entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def set_watersurface():
        global watersurface
        watersurface = float(entry.get())
        label.config(text=f"Water Surface Distance Set to: {watersurface}m")

    button = ttk.Button(input_frame, text="Set", command=set_watersurface)
    button.pack(side=tk.LEFT, padx=5)

    # Table for displaying tracked people
    columns = ("ID", "Time in Water", "Status")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)

    tree.column("ID", width=100, anchor=tk.CENTER)
    tree.column("Time in Water", width=200, anchor=tk.CENTER)
    tree.column("Status", width=150, anchor=tk.CENTER)

    tree.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    # Style the scrollbar
    vsb = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)

    def update_table():
        for row in tree.get_children():
            tree.delete(row)

        for person_id, person_data in our_Array.items():
            status = "Drowning" if person_data.isDrowning(person_data.distance, watersurface) else "Not Drowning"
            color = "red" if status == "Drowning" else "green"
            tree.insert("", "end", values=(person_id, person_data.timer.printDuration(), status), tags=(color,))

        for person_id, person_data in people_above.items():
            tree.insert("", "end", values=(person_id, 0, "Not Drowning"), tags=("green",))

        root.after(1000, update_table)  # update every second

    # Tag configurations for color changes
    tree.tag_configure('red', background='red', foreground='white')
    tree.tag_configure('green', background='green', foreground='white')

    update_table()

    def detection_loop():
        pipe = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipe.start(config)
        aligned_stream = rs.align(rs.stream.color)
        detector = YoloDetector(model_name=None)
        object_tracker = DeepSort(max_age=5,
                                  n_init=2,
                                  nms_max_overlap=1.0,
                                  max_cosine_distance=0.3,
                                  nn_budget=None,
                                  override_track_class=None,
                                  embedder="mobilenet",
                                  half=True,
                                  bgr=True,
                                  embedder_gpu=True,
                                  embedder_model_name=None,
                                  embedder_wts=None,
                                  polygon=False,
                                  today=None)

        mainFlag = False

        try:
            while True:
                frames = pipe.wait_for_frames()
                frames = aligned_stream.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                start = time.perf_counter()
                results = detector.score_frame(color_image)
                img, detections = detector.plot_boxes(results, color_image, height=color_image.shape[0], width=color_image.shape[1], confidence=0.5)
                tracks = object_tracker.update_tracks(detections, frame=img)

                # Process tracking results and update our_Array and people_above
                keys_to_remove = []
                for key in our_Array:
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        if mainFlag:
                            continue  
                        if track.track_id == key:
                            mainFlag = True
                            continue
                    if not mainFlag:
                        keys_to_remove.append(key)
                    mainFlag = False

                for k in keys_to_remove:
                    del our_Array[k]

                keys_to_remove = []
                mainFlag = False
                if len(people_above) > 0:
                    for key in people_above:
                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                            if mainFlag:
                                continue  
                            if track.track_id == key:
                                mainFlag = True
                                continue
                        if not mainFlag:
                            keys_to_remove.append(key)
                        mainFlag = False

                for k in keys_to_remove:
                    del people_above[k]

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    bbox = ltrb

                    x_center = int((bbox[0] + bbox[2]) / 2)
                    y_center = int((bbox[1] + bbox[3]) / 2)

                    keys_to_remove = []
                    keys_to_remove_from_above = []
                    flag = False

                    if 0 <= x_center < depth_frame.get_width() and 0 <= y_center < depth_frame.get_height():
                        depth = depth_frame.get_distance(x_center, y_center)
                        for key, value in our_Array.items():  # in the underwater array
                            if key == track_id:
                                flag = True
                                value.updateDistance(depth, watersurface)
                                if value.isDrowning(depth, watersurface):
                                    print(f"A person {key} is drowning at x={x_center}, y={y_center}, depth={depth-watersurface:.2f}m, help")
                                else:
                                    if value.isUnderWater:
                                        continue
                                    else:
                                        keys_to_remove.append(key)
                        if not flag:
                            if track_id in people_above:
                                people_above[track_id].updateDistance(depth, watersurface)
                            else:
                                curr_person = person_in_pool(track_id, depth, watersurface)
                                if curr_person.isUnderWater:
                                    our_Array[track_id] = curr_person
                                else:
                                    people_above[track_id] = curr_person

                        for key, value in people_above.items():
                            if key == track_id:
                                value.updateDistance(depth, watersurface)
                                if value.isDrowning(depth, watersurface):
                                    keys_to_remove_from_above.append(key)
                                else:
                                    if not value.isUnderWater:
                                        continue
                                    else:
                                        keys_to_remove_from_above.append(key)

                        for k in keys_to_remove:
                            people_above[k] = our_Array[k]

                        for k in keys_to_remove:
                            del our_Array[k]

                        for k in keys_to_remove_from_above:
                            our_Array[k] = people_above[k]

                        for k in keys_to_remove_from_above:
                            del people_above[k]

                        # Draw bounding boxes and display depth for all tracked people
                        color = (0, 0, 255) if track_id in our_Array else (255, 0, 0)
                        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.putText(img, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        cv2.putText(img, f"Depth: {depth:.2f}m", (int(bbox[0]), int(bbox[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Calculate frame processing time
                end = time.perf_counter()
                print(f"Frame processing time: {end - start:.4f} seconds")

                cv2.imshow('Drowning Detection System', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pipe.stop()
            cv2.destroyAllWindows()

    # Start the detection loop in a separate thread
    Thread(target=detection_loop, daemon=True).start()

    # Start the Tkinter main loop
    root.mainloop()
