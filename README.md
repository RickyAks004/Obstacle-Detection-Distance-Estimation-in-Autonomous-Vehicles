# 📌 Obstacle Detection & Monocular Depth Estimation in Autonomous Vehicles
Monocular obstacle detection and depth estimation using YOLOv8 and Apple's Depth-Pro on the KITTI autonomous driving dataset.

YOLOv8 + Depth-Pro | KITTI Dataset | Single RGB Camera

🚀 Overview

This project implements a real-time obstacle detection and distance estimation system using only a single RGB camera, without relying on stereo vision or dedicated depth sensors.
It combines:

* YOLOv8 for object detection
* Apple’s Depth-Pro for monocular depth estimation
* KITTI Dataset for autonomous driving benchmarking

**This lightweight pipeline is suitable for EV perception, robotics navigation, ADAS, and assistive systems.**

**🧠 Core Features**

- Single-camera obstacle perception—no stereo vision or LiDAR required
- YOLOv8-based object detection with high accuracy and real-time performance
- Depth-Pro monocular depth estimation to approximate object distance
- KITTI dataset integration for training & evaluation
- Detection–Depth fusion to estimate per-object distance from camera view
- Modular & extensible Python codebase using PyTorch and Ultralytics

**🗂️ Project Structure**

_📦 obstacle-detection-depth-estimation_
 ┣ 📁 models/           # YOLOv8 + Depth-Pro loading and utilities
 
 ┣ 📁 data/             # KITTI dataset preprocessing scripts
 
 ┣ 📁 utils/            # Visualization, depth processing, fusion modules
 
 ┣ 📁 results/          # Sample outputs, predictions, heatmaps
 
 ┣ 📜 inference.py      # Run detection + depth estimation on images/videos
 
 ┣ 📜 train.py          # Optional re-training code for YOLOv8
 
 ┗ 📜 README.md

 

**📊 Dataset – KITTI**

The KITTI Vision Benchmark Suite is used, containing:

- Road scenes from autonomous driving
- RGB images with calibration data
- Real-world obstacles: cars, pedestrians, cyclists
  
This ensures the pipeline generalizes well to EV and robotics environments.

**🖼️ Sample Outputs**

    [ Detection ] → [ Depth Map ] → [ Distance Estimation ]

**▶️ How to Run**

1️⃣ Clone the repository

    git clone https://github.com/yourusername/your-repo.git
    cd your-repo

2️⃣ Install requirements

    pip install -r requirements.txt

3️⃣ Run detection + depth estimation

    python inference.py --source sample_video.mp4

**⚙️ Technologies Used**

* YOLOv8 (Ultralytics) – Object Detection
* Depth-Pro (Apple) – Monocular Depth Estimation
* PyTorch – Model Processing
* OpenCV – Visualization & Video Processing
* KITTI Dataset – Benchmark Dataset

**🧩 Applications**

* 🟢 Autonomous Electric Vehicles
* 🟢 Autonomous Mobile Robots
* 🟢 ADAS systems
* 🟢 Obstacle-aware navigation
* 🟢 Assistive perception tools

**🌟 Future Improvements**

+ Add temporal depth smoothing using optical flow
+ Integrate LiDAR-like pseudo depth for 3D point-cloud generation
+ ONNX export for edge deployment
+ ROS2 integration

🤝 Contributing

Pull requests, issue reports, and suggestions are welcome!

📄 License

MIT License — feel free to use and modify.
