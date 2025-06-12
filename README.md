# Python Image Stitching Implementation

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

이미지 스티칭(Image Stitching)을 학습하고 이해하기 위한 교육용 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 OpenCV의 `Stitcher` 클래스를 사용하지 않고 구현한 이미지 스티칭 알고리즘입니다. OpenCV의 C/C++ 소스코드를 참조하여 주요 함수들을 Python으로 포팅했습니다.

### 주요 특징
- 🔧 OpenCV Stitcher 클래스 없이 구현
- 📚 교육 목적의 상세한 구현
- 🐍 OpenCV C/C++ 코드를 Python으로 포팅
- 📖 단계별 스티칭 파이프라인 제공

## 🚀 설치 및 실행

### 필수 요구사항
```bash
pip install -r requirements.txt
```

### 기본 사용법
```bash
python main.py --input_folder [폴더경로] [옵션들...]
```

### 명령어 옵션

| 옵션 | 설명 | 기본값 |
|------|------|---------|
| `--input_folder` | 스티칭할 이미지들이 있는 폴더 경로 | `./problem_3` |
| `--output` | 결과 이미지 파일명 | `result.jpg` |
| `--warp` | 워핑 방식 (plane, spherical, cylindrical 등) | `spherical` |
| `--feature_detector` | 특징점 검출 알고리즘 (SIFT, ORB, BRISK 등) | `SIFT` |
| `--work_megapix` | 이미지 등록 단계 해상도 (Mpx) | `0.6` |
| `--match_conf` | 특징점 매칭 신뢰도 임계값 | ORB: 0.3, 기타: 0.65 |
| `--conf_thresh` | 파노라마 이미지 신뢰도 임계값 | `1.0` |
| `--ba` | Bundle Adjustment 비용함수 (ray, reproj, affine, no) | `ray` |
| `--ba_refine_mask` | Bundle Adjustment 파라미터 마스크 | `xxxxx` |
| `--wave_correct` | 파도 효과 보정 (horiz, vert, no) | `horiz` |
| `--seam_megapix` | 심 추정 단계 해상도 (Mpx) | `0.1` |
| `--seam` | 심 추정 방법 (gc_color, dp_color 등) | `gc_color` |
| `--compose_megapix` | 합성 단계 해상도 (-1은 원본 해상도) | `-1` |
| `--expos_comp` | 노출 보상 방법 (gain_blocks, gain 등) | `gain_blocks` |
| `--expos_comp_nr_feeds` | 노출 보상 피드 수 | `1` |
| `--expos_comp_nr_filtering` | 노출 보상 필터링 반복 수 | `2` |
| `--expos_comp_block_size` | 노출 보상 블록 크기 (픽셀) | `32` |
| `--blend` | 블렌딩 방법 (multiband, feather, no) | `multiband` |
| `--blend_strength` | 블렌딩 강도 [0-100] | `5` |
| `--rangewidth` | 매칭할 이미지 수 제한 | `-1` |

### 사용 예제

#### 1. 평면 워핑으로 problem_1 폴더 스티칭
```bash
python main.py --input_folder ./problem_1 --warp plane --output result_plane.jpg
```

#### 2. 구형 워핑으로 problem_2 폴더 스티칭
```bash
python main.py --input_folder ./problem_2 --warp spherical --output result_spherical.jpg
```

#### 3. 원통형 워핑으로 problem_3 폴더 스티칭
```bash
python main.py --input_folder ./problem_3 --warp cylindrical --output result_cylindrical.jpg
```

## 📁 프로젝트 구조

```
├── main.py                 # 메인 실행 파일
├── utils.py                # 유틸리티 함수들 (핵심 구현)
├── requirements.txt        # 패키지 의존성
├── README.md              # 프로젝트 문서
├── result.jpg             # 결과 이미지 예시
├── opencv_stitching/      # OpenCV 원본 소스코드 참조
│   ├── include/           # C++ 헤더 파일들
│   └── src/              # C++ 구현 파일들
├── problem_1/            # 테스트 이미지 세트 1
├── problem_2/            # 테스트 이미지 세트 2
└── problem_3/            # 테스트 이미지 세트 3
```

## 🔧 구현 내용

### 포팅된 주요 함수들
- **Python 구현**: `utils.py`
- **원본 C++ 헤더**: `opencv_stitching/include/opencv2/stitching/detail/`
- **원본 C++ 소스**: `opencv_stitching/src/`

## 📖 학습 자료

이 프로젝트는 다음 자료를 참고하여 제작되었습니다:
- [OpenCV Stitching Tutorial](https://github.com/opencv/opencv/blob/1f674dcdb4ab57aac6883af3a37d6f45307b73af/samples/python/stitching_detailed.py)
- OpenCV 공식 문서


## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## ⚠️ 주의사항

이 프로젝트는 **학습 목적**으로 제작되었으며, 상용 프로젝트에서는 OpenCV의 공식 Stitcher 클래스 사용을 권장합니다.



