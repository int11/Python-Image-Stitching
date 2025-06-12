import numpy as np
import cv2 as cv
from collections import deque
import os


def get_features_All(images, feature_detector, work_scale=1.):
    """
    Function to extract keypoints and descriptors from each image in the image list
    
    Resize images before feature detection to improve computational efficiency.
    The work_scale parameter allows specifying the image resize ratio.
    
    Args:
        images (list): List of images to extract features from
        work_scale (float): Image resize ratio (0.0 ~ 1.0). 1.0 is original size, 0.5 is half size
    Returns:
        list: List of cv.detail.ImageFeatures objects for each image. Each object has the following attributes:
            - keypoints: List of keypoints
            - descriptors: Descriptor matrix
            - img_size: (width, height) tuple
    """
    result = []
    
    for img in images:
        if work_scale != 1:
            img = cv.resize(src=img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)

        features = cv.detail.computeImageFeatures2(feature_detector, img)
        result.append(features)
    
    return result


def match(features1, features2, match_conf=0.3):
    """
    Python function that performs the same operation as C++ CpuMatcher::match.
    
    Parameters
    ----------
    features1, features2 : cv.detail.ImageFeatures
        cv.detail.ImageFeatures objects with descriptors and keypoints attributes.
    match_conf : float
        Lowe's ratio test threshold (0~1).
    
    Returns
    -------
    matches : list
        List of cv.DMatch objects. Each DMatch object has the following attributes:
        - queryIdx: Feature point index in the first image (features1)
        - trainIdx: Feature point index in the second image (features2)  
        - distance: Similarity distance between two feature points (lower values indicate higher similarity)
    """    # Type check
    d1 = features1.descriptors
    d2 = features2.descriptors
    if isinstance(d1, cv.UMat):
        d1 = d1.get()
    if isinstance(d2, cv.UMat):
        d2 = d2.get()


    # Initialize result object
    matches = []
    seen = set()

    # Select different algorithms based on descriptor type, using ORB here. See features_finder.
    if d1.dtype == np.uint8: # When features are found with binary descriptors like ORB
        index_params = dict(algorithm=6)  # FLANN_INDEX_LSH
        search_params = dict(checks=32)   # Number of neighbors to check during search
    else:                    # When features are found with float descriptors like SIFT
        index_params = dict(algorithm=1, trees=4)  # KDTreeIndexParams
        search_params = dict(checks=32, eps=0., sorted=True)  # SearchParams

    matcher = cv.FlannBasedMatcher(index_params, search_params)
      # ---- 1->2 matching ----
    knn12 = matcher.knnMatch(d1, d2, k=2)
    for match in knn12:
        if len(match) < 2:  # Skip if less than 2, same as C++ code
            continue
        m0, m1 = match
        """ 
        Select as a good match when the similarity of the best match is less than 70% of the similarity of the second-best match
        This implements a matching filtering method called Lowe's ratio test, which ensures the uniqueness of matches

        Case 1: Selected as a good match
        m.distance = 100  # Similarity of the best match
        n.distance = 200  # Similarity of the second-best match
        100 < 0.7 * 200 (100 < 140) -> True
        Added to good_matches

        Case 2: Not selected as a good match
        m.distance = 150  # Similarity of the best match
        n.distance = 200  # Similarity of the second-best match
        150 < 0.7 * 200 (150 < 140) -> False
        Not added to good_matches

        Case 3: Very good match
        m.distance = 50   # Similarity of the best match
        n.distance = 200  # Similarity of the second-best match
        50 < 0.7 * 200 (50 < 140) -> True
        Added to good_matches

        Case 4: Very similar matches
        m.distance = 180  # Similarity of the best match
        n.distance = 200  # Similarity of the second-best match
        180 < 0.7 * 200 (180 < 140) -> False
        Not added to good_matches

        Reasons for this filtering:
        Cases 1, 3: When the best match is clearly better than the second match
        Cases 2, 4: When the best match is similar to the second match (uncertain match) 
        """
        if m0.distance < (1.0 - match_conf) * m1.distance:
            matches.append(m0)
            seen.add((m0.queryIdx, m0.trainIdx))
      # ---- 2->1 matching ----
    knn21 = matcher.knnMatch(d2, d1, k=2)
    for match in knn21:
        if len(match) < 2:  # Skip if less than 2, same as C++ code
            continue
        m0, m1 = match
        if m0.distance < (1.0 - match_conf) * m1.distance:
            # If (trainIdx, queryIdx) was not already added in 1->2
            if (m0.trainIdx, m0.queryIdx) not in seen:
                # Add with query/train flipped, same as C++ code
                dm = cv.DMatch(m0.trainIdx, m0.queryIdx, m0.distance)
                matches.append(dm)
    
    return matches


def match_all(features_list,
                       match_conf=0.3,
                       num_matches_thresh1=6,
                       num_matches_thresh2=6,
                       matches_confidence_thresh=3.0):
    """
    Python function that implements the same behavior as C++ BestOf2NearestMatcher::match.

    Parameters :
        features_list : list
            List of cv.detail.ImageFeatures objects
            Each object has the following attributes:
            - keypoints: List of keypoints
            - descriptors: Descriptor matrix
            - img_size: (width, height) tuple
        match_conf : float
            ratio test threshold (C++ match_conf_)
        num_matches_thresh1 : int
            Minimum number of matches to attempt homography (C++ num_matches_thresh1_)
        num_matches_thresh2 : int
            Minimum number of inliers to attempt refinement (inliers only re-estimation) (C++ num_matches_thresh2_)
        matches_confidence_thresh : float
            Threshold to set confidence to 0 after confidence calculation (C++ matches_confindece_thresh_)

    Returns :
        matches_info : cv.detail.MatchesInfo
            cv.detail.MatchesInfo object stores matching information between image pairs.
            Main attributes:
            - src_img_idx: Source image index
            - dst_img_idx: Target image index
            - matches: List of DMatch objects containing information about matched feature point pairs
            - H: Homography matrix (3x3)
            - inliers_mask: Inlier mask estimated by RANSAC (bool array)
            - num_inliers: Number of inliers            - confidence: Matching confidence score
    """
    n_images = len(features_list)
    # Initialize 2D list of size N x N
    all_matches = [[None for _ in range(n_images)] for _ in range(n_images)]
    
    # Perform matching for all image pairs
    for i in range(n_images):
        for j in range(n_images):
            matches_info = cv.detail.MatchesInfo()
            matches_info.src_img_idx = i
            matches_info.dst_img_idx = j
            if i==j:
                all_matches[i][j] = matches_info
                continue

            features1, features2 = features_list[i], features_list[j]
            good_matches = match(features1, features2, match_conf)
            matches_info.matches = good_matches

            # Skip if insufficient matches
            n_matches = len(matches_info.matches)
            if n_matches < num_matches_thresh1:
                all_matches[i][j] = matches_info
                continue

            # 2) Prepare point set for homography
            src_pts = []
            dst_pts = []

            w1, h1 = features1.img_size
            w2, h2 = features2.img_size
            for m in matches_info.matches:
                x1, y1 = features1.keypoints[m.queryIdx].pt
                x2, y2 = features2.keypoints[m.trainIdx].pt
                
                # Image center correction, image coordinate system typically has origin at top-left (0,0)
                # But in camera models, it's more natural to have the image center as origin
                src_pts.append([x1 - w1 * 0.5, y1 - h1 * 0.5])
                dst_pts.append([x2 - w2 * 0.5, y2 - h2 * 0.5])

            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)

            # 3) RANSAC-based homography estimation, H (Homography Matrix) is a matrix that transforms src_pts image to dst_pts image perspective
            # inliers_mask is a boolean array indicating reliable points (inliers) among the matching points used to calculate the homography matrix
            # Continue if H is None or determinant is close to 0
            H, inliers_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
            if H is None or abs(np.linalg.det(H)) < np.finfo(float).eps:
                all_matches[i][j] = matches_info
                continue

            matches_info.H = H
            inliers_mask = inliers_mask.ravel().astype(bool)
            matches_info.inliers_mask = inliers_mask.tolist()

            # 4) Calculate number of inliers & confidence
            num_inliers = int(inliers_mask.sum())
            matches_info.num_inliers = num_inliers

            # Calculate confidence (using formula proposed by M. Brown and D. Lowe's paper)
            confidence = num_inliers / (8.0 + 0.3 * float(n_matches))
            # Remove matching between images that are too close
            if confidence > matches_confidence_thresh:
                confidence = 0
            matches_info.confidence = confidence

            # 5) Refinement (re-estimation with inliers only)
            if num_inliers < num_matches_thresh2:
                all_matches[i][j] = matches_info
                continue

            # Extract only inlier points
            src_in = src_pts[inliers_mask]
            dst_in = dst_pts[inliers_mask]
            # RANSAC again with only inliers values (don't need inliers_mask so don't receive return)
            H2, _ = cv.findHomography(src_in, dst_in, cv.RANSAC)
            if H2 is not None and abs(np.linalg.det(H2)) > np.finfo(float).eps:
                matches_info.H = H2

            all_matches[i][j] = matches_info

    return all_matches


def focals_from_homography(H):
    """
    Corresponds to C++ detail::focalsFromHomography
    Focal length estimation using Image of the Absolute Conic. Requires study
    Parameters:
        H (np.ndarray): 3x3 homography matrix of type np.float64
        
    Returns:
        tuple: (f0, f1) where:
            - f0 (float or None): Estimated focal length in x direction
            - f1 (float or None): Estimated focal length in y direction
            Returns None for invalid estimates
    """
    # Map homography matrix elements to meaningful variable names
    h00, h01, h02 = H[0, 0], H[0, 1], H[0, 2]
    h10, h11, h12 = H[1, 0], H[1, 1], H[1, 2]
    h20, h21, h22 = H[2, 0], H[2, 1], H[2, 2]

    # f1 (Y direction)
    d1 = h20 * h21
    d2 = (h21 - h20) * (h21 + h20)
    v1 = -(h00 * h01 + h10 * h11) / d1
    v2 = (h00 * h00 + h10 * h10 - h01 * h01 - h11 * h11) / d2
    if v1 < v2:
        v1, v2 = v2, v1; 
        d1, d2 = d2, d1
    if v1 > 0 and v2 > 0:
        f1 = np.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0:
        f1 = np.sqrt(v1)
    else:
        f1 = None

    # f0 (X direction)
    d1 = h00 * h10 + h01 * h11
    d2 = h00 * h00 + h01 * h01 - h10 * h10 - h11 * h11
    v1 = -h02 * h12 / d1
    v2 = (h12 * h12 - h02 * h02) / d2
    if v1 < v2:
        v1, v2 = v2, v1; 
        d1, d2 = d2, d1
    if v1 > 0 and v2 > 0:
        f0 = np.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0:
        f0 = np.sqrt(v1)
    else:
        f0 = None

    return f0, f1

    
def estimate_focal(features, pairwise_matches):
    """
    Corresponds to C++ detail::estimateFocal
    """
    n = len(features)
    all_f = []
    for m in pairwise_matches:
        H = m.H
        if H is None or H.size == 0:
            continue
        f0, f1 = focals_from_homography(H.astype(np.float64))

        if f0 != None and f1 != None:
            all_f.append(np.sqrt(f0*f1))
            
    # When there are sufficient matching pairs, use only the median value among focal lengths as focal length
    if len(all_f) >= n-1:
        med = np.median(all_f)
        return med
    # fallback
    sums = sum(f.img_size[0] + f.img_size[1] for f in features)
    avg = sums / n
    return avg


class Union_Find:
    """
    Union-Find Algorithm by size
    When merging two sets in Union-Find, using the method of attaching the smaller set to the larger set
    → makes the overall tree structure less deep,
    → and improves performance.
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
    
    def getParent(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.getParent(self.parent[x])
        return self.parent[x]
    
    def merge(self, x, y):
        x_root = self.getParent(x)
        y_root = self.getParent(y)
        
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]

    def isSameParent(self, x, y):
        if self.getParent(x) == self.getParent(y):
            return True
        else:
            return False
        

def find_max_spanning_tree(num_images, pairwise_matches):
    """
    Corresponds to C++ detail::findMaxSpanningTree
    Returns
      graph: {u: [(v,weight),...], ...}
      centers: [idx1, idx2]
    """
    # weight = num_inliers and since it's maximum spanning tree, sort in descending order and use only those >= 0 as edges
    edges = sorted(pairwise_matches, key=lambda x: x.num_inliers, reverse=True)
    edges = [(i.src_img_idx, i.dst_img_idx, i.num_inliers) for i in edges if i.num_inliers > 0]

    ds = Union_Find(num_images)
    # Information of edges from each node [[connected node index], [weight], ...]
    graph = [[] for i in range(num_images)]

    # Create Maximum Spanning Tree (MST) using Kruskal's Algorithm
    for i, j, w in edges:
        if ds.isSameParent(i,j) == False:
            ds.merge(i,j)
            graph[i].append((j,w))
            graph[j].append((i,w))

    # Find leaves in tree structure
    leaves = [i for i in range(len(graph)) if len(graph[i]) == 1]
    # Calculate maximum distance from each node to all leaves
    # Example: max_dists[2] = 3 → Node 2 is at most 3 steps away from some leaf
    max_dists = [0]*num_images
    for leaf in leaves:
        dist = [-1]*num_images
        dist[leaf] = 0
        q = deque([leaf])
        while q:
            u = q.popleft()
            for v,_ in graph[u]:
                if dist[v]<0:
                    dist[v] = dist[u]+1
                    q.append(v)
        for k in range(num_images):
            max_dists[k] = max(max_dists[k], dist[k])
    # The node with the shortest distance to the farthest leaf is the "center of the tree", or "center image"
    mm = min(max_dists)
    # When the number of nodes is even, there are 2 center nodes; when odd, there is 1
    centers = [i for i,d in enumerate(max_dists) if d==mm]
    return graph, centers


def leaveBiggestComponent(features, pairwise_matches, conf_threshold):
    num_images = len(features)
    comps = Union_Find(num_images)

    # Maximum Spanning Tree (MST) using Kruskal's Algorithm
    # Merge pairs with confidence above threshold into the same component
    for m in pairwise_matches:
        i = m.src_img_idx
        j = m.dst_img_idx
        if m.confidence < conf_threshold:
            continue
        if comps.isSameParent(i,j) == False:
            comps.merge(i, j)

    # Calculate largest component index
    max_comp = max(range(num_images), key=lambda x: comps.size[x])

    # Separate indices to keep and indices to remove
    indices = []
    indices_removed = []
    for i in range(num_images):
        if comps.getParent(i) == max_comp:
            indices.append(i)
        else: 
            indices_removed.append(i) 
 
    # Reconstruct features and pairwise_matches in index order
    features_subset = [] 
    pairwise_matches_subset = [] 
    for i in indices:
        features_subset.append(features[i])
        for j in indices:
            pairwise_matches_subset.append(pairwise_matches[i * num_images + j])

    return indices


def matchesGraphAsString(img_names, pairwise_matches, conf_threshold):
    """Convert matches graph to DOT language string format.
    
    Args:
        img_names: List of image names
        pairwise_matches: List of pairwise matches between images
        conf_threshold: Confidence threshold for matches
        
    Returns:
        str: DOT language representation of the matches graph
    """
    num_images = len(img_names)
    comps = Union_Find(num_images)

    str = "matches_graph\n"
    graph = set()
      # Maximum Spanning Tree (MST) using Kruskal's Algorithm
    # Merge pairs with confidence above threshold into the same component
    for m in pairwise_matches:
        i = m.src_img_idx
        j = m.dst_img_idx
        if m.confidence < conf_threshold:
            continue
        if comps.isSameParent(i,j) == False:
            comps.merge(i, j)
            graph.add((i, j))
    
    # Add edges to graph string
    for i, j in graph:
        # Extract filenames from paths
        name_src = os.path.basename(img_names[i])
        name_dst = os.path.basename(img_names[j])
        
        pos = i * num_images + j
        match = pairwise_matches[pos]
        num_inliers = sum(1 for x in match.inliers_mask if x)
        
        str += f'"{name_src}" -- "{name_dst}"[label="Num_matches={len(match.matches)}, num_inliers={num_inliers}, confidence={match.confidence:.5f}"];\n'
    
    # Add isolated nodes
    for i in range(num_images):
        if comps.size[comps.getParent(i)] == 1:
            name = os.path.basename(img_names[i])
            str += f'"{name}";\n'
    
    return str


def propagation(graph, center_image_index, function):
    """
    Starting from the center node, visit adjacent neighbor nodes in BFS order, calling function whenever an unvisited node (V) is discovered, then adding it to the queue
    Process of propagating camera rotation matrix R in tree (graph) form

    graph: Maximum Spanning Tree (MST)
    start: center image
    function(u,v) called in BFS order
    """
    seen = {center_image_index}
    q = deque([center_image_index])
    while q:
        u = q.popleft()
        for v,_ in graph[u]:
            if v not in seen:
                function(u, v)
                seen.add(v)
                q.append(v)


def homography_based_estimate(features, pairwise_matches):
    """
    Corresponds to C++ detail::HomographyBasedEstimator::estimate
    Assumes homography matrix is pure rotation matrix, so there are errors.
    Requires optimization through Bundle Adjustment (BA).
    """
    n = len(features)
    # 1) focal estimation or principal point adjust
    focals = estimate_focal(features, pairwise_matches)

    cameras = [cv.detail.CameraParams() for _ in range(n)]
    for i in range(n): 
        cameras[i].focal = focals
        cameras[i].aspect = 1.0
        cameras[i].ppx = 0.0
        cameras[i].ppy = 0.0
        cameras[i].R = np.eye(3, dtype=np.float64)
        cameras[i].t = np.zeros((3, 1), dtype=np.float64)
        

    # 2) Maximum spanning tree graph & center image index
    graph, centers = find_max_spanning_tree(n, pairwise_matches)
    # When nodes are even, there are 2 center nodes, so select only one
    root = centers[0]

    def visit_function(u, v):
        # Function to execute when visited.
        # Corresponds to C++ CalcRotation functor
        idx = u*n + v
        m = pairwise_matches[idx]
        H = m.H.astype(np.float64)
        
        # K_from, K_to
        # Kf, Kt are camera matrices of center image and v image respectively
        cam_u = cameras[u]
        cam_v = cameras[v]
        Kf = np.array([
            [cam_u.focal,                          0, cam_u.ppx],
            [          0, cam_u.focal * cam_u.aspect, cam_u.ppy],
            [          0,                          0,         1]
        ], dtype=np.float64)

        Kt = np.array([
            [cam_v.focal,                          0, cam_v.ppx],
            [          0, cam_v.focal * cam_v.aspect, cam_v.ppy],
            [          0,                          0,         1]
        ], dtype=np.float64)

        # Here H is the homography matrix that transforms center image to v image, visiting adjacent neighbor nodes in BFS order from center node.
        # Calculate Rotation matrix that transforms all v images to center image
        # H = Kt^-1 * R * Kf        # Transformation matrix from center image to v image
        # R = Kf * H * Kt^-1        # Transformation matrix from center image to v image
        # R^-1 = Kf^-1 * H^-1 * Kt  # Transformation matrix from v image to center image
        R_edge = np.linalg.pinv(Kf) @ np.linalg.pinv(H) @ Kt
        cameras[v].R = cameras[u].R @ R_edge

    propagation(graph, root, visit_function)

    # 3) restore principal point shift if focal estimated
    for i in range(n):
        w,h = features[i].img_size
        cameras[i].ppx += 0.5*w
        cameras[i].ppy += 0.5*h

    return cameras


def cylindrical_warp(img, f_width, f_height, c_w, c_h):
    """
    Function to warp image cylindrically
    
    Args:
        img (numpy.ndarray): Input image
        f_width (float): Focal length in horizontal direction (pixel units)
        f_height (float): Focal length in vertical direction (pixel units)
        c_w (float): Image center x coordinate
        c_h (float): Image center y coordinate
        
    Returns:
        numpy.ndarray: Cylindrically warped image
    """
    K = np.array([[f_width, 0, c_w],
                  [0, f_height, c_h],
                  [0, 0, 1]])
    
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]

    theta= (x- c_w)/f_width # angle theta
    h = (y-c_h)/f_height # height
    p = np.array([np.sin(theta),h,np.cos(theta)])
    p = p.T
    p = p.reshape(-1,3)
    image_points = K.dot(p.T).T
    points = image_points[:,:-1]/image_points[:,[-1]]
    points = points.reshape(img.shape[0],img.shape[1],-1)
    cylinder = cv.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv.INTER_LINEAR)
    return cylinder