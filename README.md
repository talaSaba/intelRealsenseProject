# **Real-Time Drowning Detection Using Intel RealSense**  
## **Overview**  
This project is designed to detect potential drowning incidents in real-time using the Intel RealSense camera.  
The system measures the distance between objects and the water surface,  
and if an object remains underwater for a predefined period, it triggers an alert to notify a supervisor.  
the main components of the system are:  
#### **hardware:**  
-Intel RealSense camera (D435)  
-CPU/GPU:this code works on CPU but for better frame rate we recommend using GPU.  
#### **software:**  
this project uses python=3.10 enviroment  
-OpenCV: Install opencv in python under command:  
```python   
 $pip install opencv-python    
 ```  
-librealsense: Install librealsense python of Realsense D435 SDK by following:
```python   
$pip install pyrealsense2
```
-NumPy:Install numpy by following:
```python     
$pip install numpy
```
-torch: Install PyTorch
```python     
$pip install torch
```
for GPU usage follow this link: https://www.youtube.com/watch?v=d_jBX7OrptI&t=139s  
-Deep Sort: install deep_sort_realtime:
```python     
$pip install deep_sort_realtime
```

## **how to run the project:**  
**Setup:**  
Ensure all required packages are installed. If you're using CUDA for enhanced performance, make sure it’s properly installed and configured (optional but recommended).  
Connect the camera to your computer.  
  
**Running the Application:**  
Execute the main.py script to start the application.  

**Using the Application:**  
place the camera above the water surface. enter the distance between the camera and the water surface.  
Click the "Set" button to calibrate the system.  
You're all set! The system is now ready to monitor and help save lives. 

**video demonstrations**
https://drive.google.com/drive/folders/183NWzCz2rN-Ej-j-eLgLqGSkjmARVeEt?usp=drive_link

**References:**  
https://github.com/niconielsen32/ComputerVision  

