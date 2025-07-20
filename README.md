## Rice Bag Detection System for Industrial Automation

🧠 This project delivers a real-time **computer vision solution** for automated **detection, segmentation, and counting** of rice bags on a conveyor system, tailored for industrial packaging environments.

### 🛠️ Key Capabilities

* **Accurate object detection** with bounding boxes for tracking rice bags
* **Contour-based segmentation** to analyze shape and orientation
* **Automated counting** for throughput monitoring and reporting

### 🚀 Industrial Relevance

* Enhances **quality assurance** by identifying deformed or misaligned packages
* Supports **real-time decision-making** in high-speed production lines
* Enables **integration with PLCs and sorting systems** for full automation

### Take a look of the example outputs: 
![Description](output.png)
![Description](output2.png)

### 📁 Project Structure

```
├── videos/
│   └── rice_package.mp4          # Example input video
├── object_background_filter.py  # Basic contour detection
├── object_background_filter_rice.py  # With object tracking and counting
├── output.png                   # Output sample image
├── output2.png                  # Output sample image
└── README.md
```

---
### 🔧 Instructions

1. **Install Dependencies**

   ```bash
   pip install opencv-python numpy
   ```

2. **Edit the Video Path**

   * `object_background_filter.py`
   * `object_background_filter_rice.py` (includes advanced counting and tracking)

   Then update the `dir` variable with the correct path to your video file:

   ```python
   dir = r"videos/rice_package.mp4"
   cap = cv.VideoCapture(dir)
   ```

3. **Run the Script**

   ```bash
   python object_background_filter_rice.py
   ```

   or

   ```bash
   python object_background_filter.py
   ```

4. **View Results**

   * The script will display the video with bounding boxes, contours, and bag count.
   * An output video file (`*_out.mp4`) will be saved automatically in the project directory.

---


