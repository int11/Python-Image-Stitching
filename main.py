import argparse
from utils import *
import cv2 as cv
import numpy as np
import os
import glob
from collections import OrderedDict


EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['gc_color'] = cv.detail_GraphCutSeamFinder('COST_COLOR')
SEAM_FIND_CHOICES['gc_colorgrad'] = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

WAVE_CORRECT_CHOICES = OrderedDict()
WAVE_CORRECT_CHOICES['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
WAVE_CORRECT_CHOICES['no'] = None
WAVE_CORRECT_CHOICES['vert'] = cv.detail.WAVE_CORRECT_VERT

BLEND_CHOICES = ('multiband', 'feather', 'no',)


def get_compensator(args):
    expos_comp_type = EXPOS_COMP_CHOICES[args.expos_comp]
    expos_comp_nr_feeds = args.expos_comp_nr_feeds
    expos_comp_block_size = args.expos_comp_block_size
    # expos_comp_nr_filtering = args.expos_comp_nr_filtering
    if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
        compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator


parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='./problem_3', dest='input_folder',  
    help="Folder containing images to stitch. Will process all supported image files in the folder.",
)
parser.add_argument('--output', action='store', default='result.jpg', type=str, dest='output',
    help="The default is 'result.jpg'",
)
parser.add_argument('--warp', action='store', default='spherical', 
    choices=[
        'plane', 'spherical', 'affine', 'cylindrical', 'fisheye', 'stereographic',
        'compressedPlaneA2B1', 'compressedPlaneA1.5B1', 'compressedPlanePortraitA2B1',
        'compressedPlanePortraitA1.5B1', 'paniniA2B1', 'paniniA1.5B1',
        'paniniPortraitA2B1', 'paniniPortraitA1.5B1', 'mercator', 'transverseMercator'
    ], 
    type=str, dest='warp',
    help="Warp surface type. The default is 'spherical'."
)
parser.add_argument('--feature_detector', action='store', default='SIFT', choices=['xfeatures2d_SURF', 'ORB', 'SIFT', 'BRISK', 'AKAZE'], type=str, dest='feature_detector',
    help="Feature detection algorithm. The default is 'SIFT'."
)
parser.add_argument('--work_megapix', action='store', default=0.6, type=float, dest='work_megapix',
    help="Resolution for image registration step. The default is 0.6 Mpx"
)
parser.add_argument('--match_conf', action='store', type=float, dest='match_conf',
    help="Confidence for feature matching step. The default is 0.3 for ORB and 0.65 for other feature types."
)
parser.add_argument('--conf_thresh', action='store', default=1.0, type=float, dest='conf_thresh',
    help="Threshold for two images are from the same panorama confidence.The default is 1.0."
)
parser.add_argument('--ba', action='store', default=list(BA_COST_CHOICES.keys())[0], choices=BA_COST_CHOICES.keys(), type=str, dest='ba',
    help="Bundle adjustment cost function. The default is '%s'." % list(BA_COST_CHOICES.keys())[0]
)
parser.add_argument('--ba_refine_mask', action='store', default='xxxxx', type=str, dest='ba_refine_mask',
    help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
        "where 'x' means refine respective parameter and '_' means don't refine, "
        "and has the following format:<fx><skew><ppx><aspect><ppy>. "
        "The default mask is 'xxxxx'. "
        "If bundle adjustment doesn't support estimation of selected parameter then "
        "the respective flag is ignored."
)
parser.add_argument('--wave_correct', action='store', default=list(WAVE_CORRECT_CHOICES.keys())[0], choices=WAVE_CORRECT_CHOICES.keys(), type=str, dest='wave_correct',
    help="Perform wave effect correction. The default is '%s'" % list(WAVE_CORRECT_CHOICES.keys())[0]
)
parser.add_argument('--seam_megapix', action='store', default=0.1, type=float, dest='seam_megapix',
    help="Resolution for seam estimation step. The default is 0.1 Mpx."
)
parser.add_argument('--seam', action='store', default=list(SEAM_FIND_CHOICES.keys())[0], choices=SEAM_FIND_CHOICES.keys(), type=str, dest='seam',
    help="Seam estimation method. The default is '%s'." % list(SEAM_FIND_CHOICES.keys())[0]
)
parser.add_argument('--compose_megapix', action='store', default=-1, type=float, dest='compose_megapix',
    help="Resolution for compositing step. Use -1 for original resolution. The default is -1"
)
parser.add_argument('--expos_comp', action='store', default=list(EXPOS_COMP_CHOICES.keys())[0], choices=EXPOS_COMP_CHOICES.keys(), type=str, dest='expos_comp',
    help="Exposure compensation method. The default is '%s'." % list(EXPOS_COMP_CHOICES.keys())[0]
)
parser.add_argument('--expos_comp_nr_feeds', action='store', default=1, type=np.int32, dest='expos_comp_nr_feeds',
    help="Number of exposure compensation feed."
)
parser.add_argument('--expos_comp_nr_filtering', action='store', default=2, type=float, dest='expos_comp_nr_filtering',
    help="Number of filtering iterations of the exposure compensation gains."
)
parser.add_argument('--expos_comp_block_size', action='store', default=32, type=np.int32, dest='expos_comp_block_size',
    help="BLock size in pixels used by the exposure compensator. The default is 32."
)
parser.add_argument('--blend', action='store', default=BLEND_CHOICES[0], choices=BLEND_CHOICES, type=str, dest='blend',
    help="Blending method. The default is '%s'." % BLEND_CHOICES[0]
)
parser.add_argument('--blend_strength', action='store', default=5, type=np.int32, dest='blend_strength',
    help="Blending strength from [0,100] range. The default is 5"
)
parser.add_argument('--rangewidth', action='store', default=-1, type=int, dest='rangewidth',
    help="uses range_width to limit number of images to match with."
)


if __name__ == '__main__':
    args = parser.parse_args()
    img_folder = args.input_folder
    work_megapix = args.work_megapix
    seam_megapix = args.seam_megapix
    compose_megapix = args.compose_megapix
    conf_thresh = args.conf_thresh
    wave_correct = WAVE_CORRECT_CHOICES[args.wave_correct]
    warp_type = args.warp
    blend_type = args.blend
    feature_detector = getattr(cv, args.feature_detector).create()

    # Get all image files from the folder
    supported_formats = ('*.jpg', '*.jpeg', '*.png')
    img_names = []
    for fmt in supported_formats:
        img_names.extend(glob.glob(os.path.join(img_folder, fmt)))
    
    if not img_names:
        print(f"No image files found in {img_folder}")
        exit()
    
    print(f"Found {len(img_names)} images to stitch:")
    for img in img_names:
        print(f"  {img}")
    
    images = [cv.imread(path) for path in img_names]
    w,h = images[0].shape[1], images[0].shape[0]

    # 이미지 크기 조정을 위한 스케일 계산
    # work_megapix가 None이면 원본 크기(1.0), 아니면 지정된 메가픽셀에 맞게 스케일 조정
    work_scale = 1 if work_megapix is None else min(1.0, np.sqrt(work_megapix * 1e6 / (w * h)))
    # seam_megapix가 None이면 원본 크기(1.0), 아니면 지정된 메가픽셀에 맞게 스케일 조정
    seam_scale = 1.0 if seam_megapix is None else min(1.0, np.sqrt(seam_megapix * 1e6 / (w * h)))
    seam_work_aspect = seam_scale / work_scale

    # 모든 이미지에서 feature 구함
    features = get_features_All(images, feature_detector=feature_detector, work_scale=work_scale)
    images = [cv.resize(src=img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT) for img in images]

    # 모든 feature 간의 매칭 수행
    matches = match_all(features)
    matches = [matches[i][j] for i in range(len(images)) for j in range(len(images))]
    for i, match in enumerate(matches):
        print(f"{i:2d}, Image {match.src_img_idx} to {match.dst_img_idx}, "
              f"Num_matches: {len(match.matches):3d}, "
              f"Confidence: {match.confidence:.5f}, "
              f"Num_inliers: {sum(match.inliers_mask):2d}")


    # 매칭 그래프 구조를 문자열 형태로 출력 (디버깅 목적), Reverse Engineering cv.detail.matchesGraphAsString function
    print(matchesGraphAsString(img_names, matches, conf_thresh))
    # 가장 큰 연결된 컴포넌트(이미지 군집)를 남기고 나머지는 필터링, Reverse Engineering cv.detail.leaveBiggestComponent function
    indices = leaveBiggestComponent(features, matches, conf_thresh)
    images = [images[i] for i in indices]
    img_names = [img_names[i] for i in indices]
    num_images = len(img_names)
    print(indices)
    print(img_names)

    # 필터링했을때 이미지가 너무 적으면 중단
    if num_images < 2:
        print("Need more images")
        exit()
    
    # Reverse Engineering OpenCV cv.detail_BestOf2NearestMatcher function
    # estimate_focal : Image of the Absolute Conic 을 이용헤 homography matrix 로부터터 focal length 추정.
    # find_max_spanning_tree : 최대 신장 트리(Maximum Spanning Tree, MST)를 크루스칼 알고리즘(Kruskal's Algorithm) 이용해 찾고 중심 노드(이미지)를 찾음.
    # propagation: 중심노드드에서부터 인접한 이웃 노드를 BFS order대로 방문하며 카메라의 회전 행렬 R을 트리(그래프) 형태로 전파(propagation)
    cameras = homography_based_estimate(features, matches)

    # 회전 행렬(R)을 float32로 캐스팅하여 후속 OpenCV 연산 호환성 확보
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    # Bundle Adjustment (BA) 최적화 객체 생성 및 설정
    adjuster = BA_COST_CHOICES[args.ba]()  # 'ray', 'reproj', 'affine', 'no' 등 선택
    adjuster.setConfThresh(conf_thresh)    # 일치도 임계값 설정

    # BA에서 어떤 파라미터를 최적화할지 결정하는 마스크 생성
    refine_mask = np.zeros((3, 3), np.uint8)
    # ba_refine_mask 문자열 예: 'x_xxx'. 'x'는 최적화, '_'는 고정
    # (0,0): focal length, (0,1): skew/aspect, (0,2): principal point,
    # (1,1): rotation, (1,2): translation
    refine_indices = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    for i, (r, c) in enumerate(refine_indices):
        if args.ba_refine_mask[i] == 'x':
            refine_mask[r, c] = 1
    adjuster.setRefinementMask(refine_mask)

    # BA 최적화 수행: features, match 정보 p, 초기 카메라 파라미터
    success, cameras = adjuster.apply(features, matches, cameras)
    if not success:
        print("Camera parameters adjusting failed.")
        exit()

    # 최적화된 카메라의 focal 값만 추출하여 중앙값 기반의 warped image scale 결정
    focals = sorted([cam.focal for cam in cameras])
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        mid = len(focals) // 2
        warped_image_scale = (focals[mid] + focals[mid - 1]) / 2

    # Wave correction (수평/수직 보정) 적용
    if wave_correct is not None:
        # 각 카메라의 회전 행렬 복사
        rmats = [np.copy(cam.R) for cam in cameras]
        # 보정된 회전 행렬 획득
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        # 원본 카메라 객체에 보정된 회전 행렬 적용
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]

    # 이후 이미지 워핑 및 마스크 생성 준비
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []

    # 각 이미지에 대한 binary 마스크 생성 (전부 흰색)
    for img in images:
        mask = cv.UMat(255 * np.ones((img.shape[0], img.shape[1]), np.uint8))
        masks.append(mask)

    # PyRotationWarper 객체 생성 (워핑 방식 및 스케일 지정)
    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)
    for idx, img in enumerate(images):
        # 카메라 내부 파라미터(K) 가져와 스케일 적용
        K = cameras[idx].K().astype(np.float32)
        K[0, 0] *= seam_work_aspect
        K[0, 2] *= seam_work_aspect
        K[1, 1] *= seam_work_aspect
        K[1, 2] *= seam_work_aspect

        # 이미지 워핑: 회전 행렬 R과 K를 사용하여 warped image, 그리고 offset(corner) 반환
        corner, image_wp = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))  # (width, height)
        images_warped.append(image_wp)

        # 마스크도 동일하게 워핑 (이진 마스크이므로 INTER_NEAREST)
        _, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())  # UMat -> numpy 배열로 변환

    # float32 타입으로 변환된 warped images 리스트 생성 (seam finder에 사용)
    images_warped_f = [img.astype(np.float32) for img in images_warped]

    # Exposure compensator 객체 가져와 feed (보정 적용 준비)
    compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    # Seam finder 적용: 워핑된 이미지와 마스크를 인자로 seam 마스크 계산
    seam_finder = SEAM_FIND_CHOICES[args.seam]
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)

    # Composition 단계 준비
    compose_scale = 1.0
    corners = []
    sizes = []
    blender = None

    # compose_scale 계산 및 warper, 카메라 파라미터 보정
    if compose_megapix > 0:
        compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (w * h)))
    compose_work_aspect = compose_scale / work_scale
    warped_image_scale *= compose_work_aspect
    warper = cv.PyRotationWarper(warp_type, warped_image_scale)
    # 각 카메라 K, focal, principal point 좌표를 compose 스케일에 맞게 업데이트
    for cam in cameras:
        cam.focal *= compose_work_aspect
        cam.ppx *= compose_work_aspect
        cam.ppy *= compose_work_aspect
    # 각 이미지 ROI(Region of Interest) 및 크기 계산
    for cam in cameras:
        K = cam.K().astype(np.float32)
        roi = warper.warpRoi((int(round(w * compose_scale)), int(round(h * compose_scale))), K, cam.R)
        corners.append(roi[0:2])
        sizes.append(roi[2:4])

        
    # 최종 compositing 단계: 원본 이미지를 사용하여 블렌딩
    for idx, name in enumerate(img_names):
        full_img = cv.imread(name)

        # compose_scale이 1이 아닌 경우 이미지를 리사이즈
        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(full_img, None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img

        # 워핑 수행
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        # 워핑된 마스크 생성
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        _, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)

        # Exposure compensation 적용
        compensator.apply(idx, corners[idx], image_warped, mask_warped)

        # int16 타입으로 변환 후 블렌더에 feed
        image_warped_s = image_warped.astype(np.int16)
        # 탐지된 seam 영역 확대하여 블렌딩 마스크 생성
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), interpolation=cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)

        # Blender 객체가 아직 생성되지 않았다면 최초 생성 및 준비
        if blender is None:
            # 최종 결과 영역 계산
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * args.blend_strength / 100
            if blend_width < 1 or blend_type == "no":
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                num_bands = int(np.log2(blend_width) - 1)
                blender.setNumBands(max(num_bands, 1))
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1.0 / blend_width)
            blender.prepare(dst_sz)

        # 블렌더에 워핑된 이미지, 마스크, 좌표 전달
        blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

    # 모든 이미지를 블렌딩하여 최종 파노라마 획득
    result, result_mask = blender.blend(None, None)

    # 결과 영상 저장 및 화면에 표시
    cv.imwrite(args.output, result)
    zoom_x = 600.0 / result.shape[1]
    dst = cv.normalize(result, None, alpha=255.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    dst = cv.resize(dst, None, fx=zoom_x, fy=zoom_x)
    cv.imshow(args.output, dst)
    cv.waitKey()
    print("Done")
    cv.destroyAllWindows()