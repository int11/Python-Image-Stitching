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
    w,h = images[0].shape[1], images[0].shape[0]    # Calculate scale for image resizing
    # If work_megapix is None, use original size (1.0), otherwise scale to specified megapixels
    work_scale = 1 if work_megapix is None else min(1.0, np.sqrt(work_megapix * 1e6 / (w * h)))
    # If seam_megapix is None, use original size (1.0), otherwise scale to specified megapixels
    seam_scale = 1.0 if seam_megapix is None else min(1.0, np.sqrt(seam_megapix * 1e6 / (w * h)))
    seam_work_aspect = seam_scale / work_scale    # Extract features from all images
    features = get_features_All(images, feature_detector=feature_detector, work_scale=work_scale)
    images = [cv.resize(src=img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT) for img in images]

    # Perform matching between all features
    matches = match_all(features)
    matches = [matches[i][j] for i in range(len(images)) for j in range(len(images))]
    for i, match in enumerate(matches):
        print(f"{i:2d}, Image {match.src_img_idx} to {match.dst_img_idx}, "
              f"Num_matches: {len(match.matches):3d}, "
              f"Confidence: {match.confidence:.5f}, "
              f"Num_inliers: {sum(match.inliers_mask):2d}")
    # Output matching graph structure as string (for debugging), Reverse Engineering cv.detail.matchesGraphAsString function
    print(matchesGraphAsString(img_names, matches, conf_thresh))
    # Filter to keep the largest connected component (image cluster), Reverse Engineering cv.detail.leaveBiggestComponent function
    indices = leaveBiggestComponent(features, matches, conf_thresh)
    images = [images[i] for i in indices]
    img_names = [img_names[i] for i in indices]
    num_images = len(img_names)
    print(indices)
    print(img_names)    # Stop if too few images after filtering
    if num_images < 2:
        print("Need more images")
        exit()
    
    # Reverse Engineering OpenCV cv.detail_BestOf2NearestMatcher function
    # estimate_focal : Estimate focal length from homography matrix using Image of the Absolute Conic
    # find_max_spanning_tree : Find Maximum Spanning Tree (MST) using Kruskal's Algorithm and find center node (image)
    # propagation: Propagate camera rotation matrix R from center node to adjacent neighbor nodes in BFS order in tree (graph) structure
    cameras = homography_based_estimate(features, matches)    # Cast rotation matrices (R) to float32 for subsequent OpenCV operation compatibility
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    # Create and configure Bundle Adjustment (BA) optimization object
    adjuster = BA_COST_CHOICES[args.ba]()  # Select from 'ray', 'reproj', 'affine', 'no'
    adjuster.setConfThresh(conf_thresh)    # Set confidence threshold

    # Create mask to determine which parameters to optimize in BA
    refine_mask = np.zeros((3, 3), np.uint8)
    # ba_refine_mask string example: 'x_xxx'. 'x' means optimize, '_' means fix
    # (0,0): focal length, (0,1): skew/aspect, (0,2): principal point,
    # (1,1): rotation, (1,2): translation
    refine_indices = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    for i, (r, c) in enumerate(refine_indices):
        if args.ba_refine_mask[i] == 'x':
            refine_mask[r, c] = 1
    adjuster.setRefinementMask(refine_mask)    # Perform BA optimization: features, match information, initial camera parameters
    success, cameras = adjuster.apply(features, matches, cameras)
    if not success:
        print("Camera parameters adjusting failed.")
        exit()

    # Extract only focal values from optimized cameras to determine warped image scale based on median
    focals = sorted([cam.focal for cam in cameras])
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        mid = len(focals) // 2
        warped_image_scale = (focals[mid] + focals[mid - 1]) / 2

    # Apply wave correction (horizontal/vertical correction)
    if wave_correct is not None:
        # Copy rotation matrix of each camera
        rmats = [np.copy(cam.R) for cam in cameras]
        # Get corrected rotation matrices
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        # Apply corrected rotation matrices to original camera objects
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]

    # Prepare for subsequent image warping and mask generation
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []    # Generate binary mask for each image (all white)
    for img in images:
        mask = cv.UMat(255 * np.ones((img.shape[0], img.shape[1]), np.uint8))
        masks.append(mask)

    # Create PyRotationWarper object (specify warping method and scale)
    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)
    for idx, img in enumerate(images):
        # Get camera intrinsic parameters (K) and apply scale
        K = cameras[idx].K().astype(np.float32)
        K[0, 0] *= seam_work_aspect
        K[0, 2] *= seam_work_aspect
        K[1, 1] *= seam_work_aspect
        K[1, 2] *= seam_work_aspect

        # Image warping: return warped image and offset (corner) using rotation matrix R and K
        corner, image_wp = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))  # (width, height)
        images_warped.append(image_wp)

        # Warp mask similarly (use INTER_NEAREST for binary mask)
        _, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())  # Convert UMat -> numpy array

    # Generate list of warped images converted to float32 type (for use in seam finder)
    images_warped_f = [img.astype(np.float32) for img in images_warped]

    # Get exposure compensator object and feed (prepare for compensation application)
    compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    # Apply seam finder: calculate seam mask using warped images and masks as arguments
    seam_finder = SEAM_FIND_CHOICES[args.seam]
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)    # Composition stage preparation
    compose_scale = 1.0
    corners = []
    sizes = []
    blender = None

    # Calculate compose_scale and correct warper, camera parameters
    if compose_megapix > 0:
        compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (w * h)))
    compose_work_aspect = compose_scale / work_scale
    warped_image_scale *= compose_work_aspect
    warper = cv.PyRotationWarper(warp_type, warped_image_scale)
    # Update each camera K, focal, principal point coordinates to match compose scale
    for cam in cameras:
        cam.focal *= compose_work_aspect
        cam.ppx *= compose_work_aspect
        cam.ppy *= compose_work_aspect
    # Calculate each image ROI (Region of Interest) and size
    for cam in cameras:
        K = cam.K().astype(np.float32)
        roi = warper.warpRoi((int(round(w * compose_scale)), int(round(h * compose_scale))), K, cam.R)
        corners.append(roi[0:2])
        sizes.append(roi[2:4])

        
    # Final compositing stage: blend using original images
    for idx, name in enumerate(img_names):
        full_img = cv.imread(name)        # Resize image if compose_scale is not 1
        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(full_img, None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img

        # Perform warping
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        # Generate warped mask
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        _, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)

        # Apply exposure compensation
        compensator.apply(idx, corners[idx], image_warped, mask_warped)

        # Convert to int16 type and feed to blender
        image_warped_s = image_warped.astype(np.int16)
        # Expand detected seam area to create blending mask
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), interpolation=cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)

        # Create and prepare blender object if not yet created
        if blender is None:
            # Calculate final result area
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

        # Pass warped image, mask, coordinates to blender
        blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

    # Blend all images to obtain final panorama
    result, result_mask = blender.blend(None, None)

    # Save result image and display on screen
    cv.imwrite(args.output, result)
    zoom_x = 600.0 / result.shape[1]
    dst = cv.normalize(result, None, alpha=255.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    dst = cv.resize(dst, None, fx=zoom_x, fy=zoom_x)
    cv.imshow(args.output, dst)
    cv.waitKey()
    print("Done")
    cv.destroyAllWindows()