[⬅️ back to top repo](https://github.com/laplamgor/kantai3d)

# Kantai3D Depth Map Generator
This is a Python script to generate depth maps for Kantai3D. It basically generates an inverted depth map in red channel using [Apple's Depth Pro model](https://github.com/apple/ml-depth-pro), and detech boob size and boob area using [Anzhc's YOLO models](https://huggingface.co/Anzhc/Anzhcs_YOLOs) and put in blue channel. It also detects face distance to adjust the depth offset so that the character's face is always around the focal plane.


# How to run
1. Install [Depth Pro](https://github.com/apple/ml-depth-pro)
2. Download [Anzhc Face seg 640 v3 y11n.pt](https://huggingface.co/Anzhc/Anzhcs_YOLOs/blob/main/Anzhc%20Face%20seg%20640%20v3%20y11n.pt) and rename to `./checkpoints/FaceSeg.pt` 
3. Download [Anzhc Breasts Seg v1 1024m.pt](https://huggingface.co/Anzhc/Anzhcs_YOLOs/blob/main/Anzhc%20Breasts%20Seg%20v1%201024m.pt) and rename to `./checkpoints/BreastSeg.pt` 
4. Download [Anzhcs Breast Size det cls v8 y11m.pt](https://huggingface.co/Anzhc/Anzhcs_YOLOs/blob/main/Anzhcs%20Breast%20size%20det%20cls%20v8%20640%20y11m.pt) and rename to `./checkpoints/BreastSize.pt`
6. Put input CGs in `./input` folder
7. Run `python run.py` 
5. Install all missing Python dependencies yourself :)


# Requirement and Performance
This extension is only tested to be compatible with
* A CUDA-supported GPU
* Python 3.12
* torch 2.7.0+cu128
On my PC (using one NVIDIA GeForce RTX 4070 Ti), it generates at ~40fps.

# Disclaimers
Kantai3D is not an officially approved program. Using it may be against the Terms of Service of DMM.

Kantai3D modifies your local game client (main.js) to achieve the visual effects, but does not modify any in-game API requests and responses. It does not affect your normal gameplay or game balance. 

Please use it at your own risk. 


This mod alone does not include any data from the original game. All custom depth maps used are 100% hand-drawn. AI-generated results are, however, not included in the scope of this open-source project due to potential copyright issues.

If you have any questions or queries, please contact me at laplamgor@gmail.com
