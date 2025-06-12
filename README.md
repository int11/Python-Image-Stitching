# Python Image Stitching Implementation

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

An educational project for learning and understanding Image Stitching.

## 📋 Project Overview

This project implements an image stitching algorithm without using OpenCV's `Stitcher` class. It ports key functions from OpenCV's C/C++ source code to Python.

### Key Features
- 🔧 Implementation without OpenCV Stitcher class
- 📚 Detailed implementation for educational purposes
- 🐍 Porting OpenCV C/C++ code to Python
- 📖 Step-by-step stitching pipeline

## 🚀 Installation and Execution

### Requirements
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
python main.py --input_folder [folder_path] [options...]
```

### Command Options

| Option | Description | Default |
|------|------|---------|
| `--input_folder` | Path to folder containing images to stitch | `./problem_3` |
| `--output` | Output image filename | `result.jpg` |
| `--warp` | Warping method (plane, spherical, cylindrical, etc.) | `spherical` |
| `--feature_detector` | Feature detection algorithm (SIFT, ORB, BRISK, etc.) | `SIFT` |
| `--work_megapix` | Resolution for image registration step (Mpx) | `0.6` |
| `--match_conf` | Feature matching confidence threshold | ORB: 0.3, Others: 0.65 |
| `--conf_thresh` | Panorama image confidence threshold | `1.0` |
| `--ba` | Bundle Adjustment cost function (ray, reproj, affine, no) | `ray` |
| `--ba_refine_mask` | Bundle Adjustment parameter mask | `xxxxx` |
| `--wave_correct` | Wave effect correction (horiz, vert, no) | `horiz` |
| `--seam_megapix` | Resolution for seam estimation step (Mpx) | `0.1` |
| `--seam` | Seam estimation method (gc_color, dp_color, etc.) | `gc_color` |
| `--compose_megapix` | Resolution for composition step (-1 for original resolution) | `-1` |
| `--expos_comp` | Exposure compensation method (gain_blocks, gain, etc.) | `gain_blocks` |
| `--expos_comp_nr_feeds` | Number of exposure compensation feeds | `1` |
| `--expos_comp_nr_filtering` | Number of exposure compensation filtering iterations | `2` |
| `--expos_comp_block_size` | Exposure compensation block size (pixels) | `32` |
| `--blend` | Blending method (multiband, feather, no) | `multiband` |
| `--blend_strength` | Blending strength [0-100] | `5` |
| `--rangewidth` | Limit number of images to match | `-1` |

### Usage Examples

#### 1. Stitching problem_1 folder with planar warping
```bash
python main.py --input_folder ./problem_1 --warp plane --output result_plane.jpg
```

#### 2. Stitching problem_2 folder with spherical warping
```bash
python main.py --input_folder ./problem_2 --warp spherical --output result_spherical.jpg
```

#### 3. Stitching problem_3 folder with cylindrical warping
```bash
python main.py --input_folder ./problem_3 --warp cylindrical --output result_cylindrical.jpg
```

## 📁 Project Structure

```
├── main.py                 # Main execution file
├── utils.py                # Utility functions (core implementation)
├── requirements.txt        # Package dependencies
├── README.md              # Project documentation
├── result.jpg             # Example result image
├── opencv_stitching/      # OpenCV original source code reference
│   ├── include/           # C++ header files
│   └── src/              # C++ implementation files
├── problem_1/            # Test image set 1
├── problem_2/            # Test image set 2
└── problem_3/            # Test image set 3
```

## 🔧 Implementation Details

### Key Ported Functions
- **Python Implementation**: `utils.py`
- **Original C++ Headers**: `opencv_stitching/include/opencv2/stitching/detail/`
- **Original C++ Source**: `opencv_stitching/src/`

## 📖 Learning Resources

This project was created with reference to the following materials:
- [OpenCV Stitching Tutorial](https://github.com/opencv/opencv/blob/1f674dcdb4ab57aac6883af3a37d6f45307b73af/samples/python/stitching_detailed.py)
- OpenCV Official Documentation

## ⚠️ Disclaimer

This project was created for **learning purposes**, and for commercial projects, it is recommended to use OpenCV's official Stitcher class.



