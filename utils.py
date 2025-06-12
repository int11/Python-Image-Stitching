import numpy as np
import cv2 as cv
from collections import deque
import os


def get_features_All(images, feature_detector, work_scale=1.):
    """
    이미지 리스트에서 각 이미지의 특징점과 디스크립터를 추출하는 함수
    
    특징점 검출 전에 이미지 크기를 조정하여 계산 효율성을 높입니다.
    work_scale 파라미터로 이미지 크기 조정 비율을 지정할 수 있습니다.
    
    Args:
        images (list): 특징점을 추출할 이미지들의 리스트
        work_scale (float): 이미지 크기 조정 비율 (0.0 ~ 1.0). 1.0은 원본 크기, 0.5는 절반 크기
    Returns:
        list: 각 이미지의 cv.detail.ImageFeatures 객체 리스트. 각 객체는 다음 속성을 가짐:
            - keypoints: 특징점 리스트
            - descriptors: 디스크립터 행렬
            - img_size: (width, height) 튜플
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
    C++ CpuMatcher::match와 동일 동작을 하는 Python 함수.
    
    Parameters
    ----------
    features1, features2 : cv.detail.ImageFeatures
        cv.detail.ImageFeatures 객체. descriptors와 keypoints 속성을 가짐.
    match_conf : float
        Lowe's ratio test 임계치(0~1).
    
    Returns
    -------
    matches : list
        cv.DMatch 객체들의 리스트. 각 DMatch 객체는 다음 속성을 가집니다:
        - queryIdx: 첫 번째 이미지(features1)의 특징점 인덱스
        - trainIdx: 두 번째 이미지(features2)의 특징점 인덱스  
        - distance: 두 특징점 간의 유사도 거리 (값이 작을수록 더 유사)
    """
    # 타입 체크
    d1 = features1.descriptors
    d2 = features2.descriptors
    if isinstance(d1, cv.UMat):
        d1 = d1.get()
    if isinstance(d2, cv.UMat):
        d2 = d2.get()


    # 결과 객체 초기화
    matches = []
    seen = set()

    # 디스크립터 타입에 따라 다른 알고리즘 선택, 여기선 ORB쓰고있음. features_finder 참고.
    if d1.dtype == np.uint8: # features를 ORB 등 binary descriptor로 찾은 경우
        index_params = dict(algorithm=6)  # FLANN_INDEX_LSH
        search_params = dict(checks=32)   # 검색 시 체크할 이웃 수
    else:                    # features를 SIFT 등 float descriptor로 찾은 경우
        index_params = dict(algorithm=1, trees=4)  # KDTreeIndexParams
        search_params = dict(checks=32, eps=0., sorted=True)  # SearchParams

    matcher = cv.FlannBasedMatcher(index_params, search_params)
    
    # ---- 1->2 매칭 ----
    knn12 = matcher.knnMatch(d1, d2, k=2)
    for match in knn12:
        if len(match) < 2:  # C++ 코드와 동일하게 2개 미만이면 건너뛰기
            continue
        m0, m1 = match
        """ 
        가장 좋은 매칭의 유사도가 두 번째로 좋은 매칭의 유사도의 70% 미만일 때 좋은 매칭점으로로 선택 
        Lowe's ratio test라고 불리는 매칭 필터링 방법을 구현한 것, 이는 매칭의 유일성(uniqueness)을 보장하는 방법

        Case 1: 좋은 매칭으로 선택되는 경우
        m.distance = 100  # 가장 좋은 매칭의 유사도
        n.distance = 200  # 두 번째로 좋은 매칭의 유사도
        100 < 0.7 * 200 (100 < 140) -> True
        good_matches에 추가됨

        Case 2: 좋은 매칭으로 선택되지 않는 경우
        m.distance = 150  # 가장 좋은 매칭의 유사도
        n.distance = 200  # 두 번째로 좋은 매칭의 유사도
        150 < 0.7 * 200 (150 < 140) -> False
        good_matches에 추가되지 않음

        Case 3: 매우 좋은 매칭
        m.distance = 50   # 가장 좋은 매칭의 유사도
        n.distance = 200  # 두 번째로 좋은 매칭의 유사도
        50 < 0.7 * 200 (50 < 140) -> True
        good_matches에 추가됨

        Case 4: 매우 유사한 매칭들
        m.distance = 180  # 가장 좋은 매칭의 유사도
        n.distance = 200  # 두 번째로 좋은 매칭의 유사도
        180 < 0.7 * 200 (180 < 140) -> False
        good_matches에 추가되지 않음

        이렇게 필터링하는 이유:
        Case 1, 3: 가장 좋은 매칭이 두 번째 매칭보다 확실히 더 좋은 경우
        Case 2, 4: 가장 좋은 매칭이 두 번째 매칭과 비슷한 경우 (불확실한 매칭) 
        """
        if m0.distance < (1.0 - match_conf) * m1.distance:
            matches.append(m0)
            seen.add((m0.queryIdx, m0.trainIdx))
    
    # ---- 2->1 매칭 ----
    knn21 = matcher.knnMatch(d2, d1, k=2)
    for match in knn21:
        if len(match) < 2:  # C++ 코드와 동일하게 2개 미만이면 건너뛰기
            continue
        m0, m1 = match
        if m0.distance < (1.0 - match_conf) * m1.distance:
            # (trainIdx, queryIdx)가 이미 1->2에서 추가되지 않았다면
            if (m0.trainIdx, m0.queryIdx) not in seen:
                # C++ 코드와 동일하게 query/train을 뒤집어 추가
                dm = cv.DMatch(m0.trainIdx, m0.queryIdx, m0.distance)
                matches.append(dm)
    
    return matches


def match_all(features_list,
                       match_conf=0.3,
                       num_matches_thresh1=6,
                       num_matches_thresh2=6,
                       matches_confidence_thresh=3.0):
    """
    C++ BestOf2NearestMatcher::match 와 동작을 동일하게 구현한 Python 함수.

    Parameters :
        features_list : list
            cv.detail.ImageFeatures 객체들의 리스트
            각 객체는 다음 속성을 가집니다:
            - keypoints: 특징점 리스트
            - descriptors: 디스크립터 행렬
            - img_size: (width, height) 튜플
        match_conf : float
            ratio test 임계치 (C++ match_conf_)
        num_matches_thresh1 : int
            호모그래피를 시도할 최소 매칭 개수 (C++ num_matches_thresh1_)
        num_matches_thresh2 : int
            리파인(인라이어만 재추정) 시도할 최소 인라이어 개수 (C++ num_matches_thresh2_)
        matches_confidence_thresh : float
            신뢰도 계산 후 0으로 만들 문턱값 (C++ matches_confindece_thresh_)

    Returns :
        matches_info : cv.detail.MatchesInfo
            cv.detail.MatchesInfo 객체는 이미지 쌍 간의 매칭 정보를 저장하는 클래스입니다.
            주요 속성:
            - src_img_idx: 소스 이미지 인덱스
            - dst_img_idx: 대상 이미지 인덱스
            - matches: DMatch 객체 리스트로, 매칭된 특징점 쌍의 정보를 담음
            - H: 호모그래피 행렬 (3x3)
            - inliers_mask: RANSAC으로 추정된 인라이어 마스크 (bool 배열)
            - num_inliers: 인라이어 개수
            - confidence: 매칭 신뢰도 점수
    """
    n_images = len(features_list)
    # N x N 크기의 2차원 리스트 초기화
    all_matches = [[None for _ in range(n_images)] for _ in range(n_images)]
    
    # 모든 이미지 쌍에 대해 매칭 수행
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

            # 충분한 매칭이 없는 경우 스킵
            n_matches = len(matches_info.matches)
            if n_matches < num_matches_thresh1:
                all_matches[i][j] = matches_info
                continue

            # 2) 호모그래피용 점셋 준비
            src_pts = []
            dst_pts = []

            w1, h1 = features1.img_size
            w2, h2 = features2.img_size

            for m in matches_info.matches:
                x1, y1 = features1.keypoints[m.queryIdx].pt
                x2, y2 = features2.keypoints[m.trainIdx].pt
                
                # 이미지 중심 보정, 이미지의 좌표계는 일반적으로 좌상단(0,0)이 원점 
                # 하지만 카메라 모델에서는 이미지의 중심을 원점으로 하는 것이 더 자연스럽러움
                src_pts.append([x1 - w1 * 0.5, y1 - h1 * 0.5])
                dst_pts.append([x2 - w2 * 0.5, y2 - h2 * 0.5])

            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)

            # 3) RANSAC 기반 호모그래피 추정, H(Homography Matrix)는 src_pts 이미지를 dst_pts 이미지의 관점으로 변환하는 행렬
            # inliers_mask는 호모그래피 행렬을 계산할 때 사용된 매칭점들 중에서 신뢰할 수 있는 점들(inliers)을 표시하는 불리언 배열
            # H is None 이거나 행렬식이 0에 가까우면 continue
            H, inliers_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
            if H is None or abs(np.linalg.det(H)) < np.finfo(float).eps:
                all_matches[i][j] = matches_info
                continue

            matches_info.H = H
            inliers_mask = inliers_mask.ravel().astype(bool)
            matches_info.inliers_mask = inliers_mask.tolist()

            # 4) 인라이어 개수 & confidence 계산
            num_inliers = int(inliers_mask.sum())
            matches_info.num_inliers = num_inliers

            # 신뢰도 계산 (M. Brown and D. Lowe의 논문에서 제안한 공식 사용)
            confidence = num_inliers / (8.0 + 0.3 * float(n_matches))
            # 너무 가까운 이미지 간의 매칭은 제거
            if confidence > matches_confidence_thresh:
                confidence = 0
            matches_info.confidence = confidence

            # 5) 리파인 (인라이어만 재추정)
            if num_inliers < num_matches_thresh2:
                all_matches[i][j] = matches_info
                continue

            # 인라이어 점만 추출
            src_in = src_pts[inliers_mask]
            dst_in = dst_pts[inliers_mask]
            # inliers 값들로만 다시 RANSAC (inliers_mask는 필요 없으므로 반환 안 받음)
            H2, _ = cv.findHomography(src_in, dst_in, cv.RANSAC)
            if H2 is not None and abs(np.linalg.det(H2)) > np.finfo(float).eps:
                matches_info.H = H2

            all_matches[i][j] = matches_info

    return all_matches


def focals_from_homography(H):
    """
    C++ detail::focalsFromHomography 에 대응
    Image of the Absolute Conic 을 이용한 focal length 추정. 공부필요
    Parameters:
        H (np.ndarray): 3x3 homography matrix of type np.float64
        
    Returns:
        tuple: (f0, f1) where:
            - f0 (float or None): Estimated focal length in x direction
            - f1 (float or None): Estimated focal length in y direction
            Returns None for invalid estimates
    """
    # 호모그래피 행렬 요소를 의미 있는 변수명으로 매핑
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
    C++ detail::estimateFocal 대응
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
            
    # 충분한 매칭 쌍이 있을때 focal length 들 중 median 값 만 focal length 로
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
    Union-Find에서 두 집합을 병합할 때, 작은 집합을 큰 집합에 붙이는 방식을 사용하면
    → 전체 트리 구조가 덜 깊어지고,
    → 성능이 좋아집니다.
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
    C++ detail::findMaxSpanningTree 대응
    반환
      graph: {u: [(v,weight),...], ...}
      centers: [idx1, idx2]
    """
    # weight = num_inliers 이고 최대 신장 트리아므로 큰 순서대로 정렬 하고 0 이상인 것만 간선으로
    edges = sorted(pairwise_matches, key=lambda x: x.num_inliers, reverse=True)
    edges = [(i.src_img_idx, i.dst_img_idx, i.num_inliers) for i in edges if i.num_inliers > 0]

    ds = Union_Find(num_images)
    # 각 노드에서 간선의 정보 [[연결된 노드 index], [weight], ...]
    graph = [[] for i in range(num_images)]

    # 최대 신장 트리(Maximum Spanning Tree, MST) 를 만들기, 크루스칼 알고리즘(Kruskal's Algorithm) 이용
    for i, j, w in edges:
        if ds.isSameParent(i,j) == False:
            ds.merge(i,j)
            graph[i].append((j,w))
            graph[j].append((i,w))

    # 트리구조에서 잎(leaf) 찾기
    leaves = [i for i in range(len(graph)) if len(graph[i]) == 1]
    # 각 노드가 모든 잎으로부터의 거리 중 최댓값을 계산
    # 예: max_dists[2] = 3 → 노드 2는 어떤 잎에서 최대 3칸 떨어짐
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
    # 가장 먼 잎과의 거리가 가장 짧은 노드가 바로 "트리의 중심", 또는 "중심 이미지"
    mm = min(max_dists)
    # 노드가 짝수일때 중심 노드는 2개 홀수일때 1개
    centers = [i for i,d in enumerate(max_dists) if d==mm]
    return graph, centers


def leaveBiggestComponent(features, pairwise_matches, conf_threshold):
    num_images = len(features)
    comps = Union_Find(num_images)

    # 최대 신장 트리(Maximum Spanning Tree, MST), 크루스칼 알고리즘(Kruskal's Algorithm) 이용
    # confidence가 임계치 이상인 쌍을 같은 컴포넌트로 병합
    for m in pairwise_matches:
        i = m.src_img_idx
        j = m.dst_img_idx
        if m.confidence < conf_threshold:
            continue
        if comps.isSameParent(i,j) == False:
            comps.merge(i, j)

    # 가장 큰 컴포넌트 인덱스 계산
    max_comp = max(range(num_images), key=lambda x: comps.size[x])

    # 남길 이미지 인덱스와 제거할 인덱스 분리
    indices = []
    indices_removed = []
    for i in range(num_images):
        if comps.getParent(i) == max_comp:
            indices.append(i)
        else: 
            indices_removed.append(i) 
 
    # features 및 pairwise_matches를 인덱스 순으로 재구성 
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
    
    # 최대 신장 트리(Maximum Spanning Tree, MST), 크루스칼 알고리즘(Kruskal's Algorithm) 이용
    # confidence가 임계치 이상인 쌍을 같은 컴포넌트로 병합
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
    중심노드드에서부터, 인접한 이웃 노드를 BFS order대로 방문하며, 방문하지 않은 노드(V)를 발견할 때마다 function를 호출한 뒤 큐에 넣음
    카메라의 회전 행렬 R을 트리(그래프) 형태로 전파(propagation)하는 과정

    graph: 최대 신장 트리(Maximum Spanning Tree, MST)
    start: 중심이미지지
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
    C++ detail::HomographyBasedEstimator::estimate 대응
    호모그래피 행렬을 순수한 회전 행렬이라고 가정하고 계산하므로 오차있음.
    Bundle Adjustment (BA)를 통해 최적화 필요.
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
        

    # 2) 최대신장트리 그래프 & 센터 이미지 index
    graph, centers = find_max_spanning_tree(n, pairwise_matches)
    # 노드가 짝수일때 중심 노드는 2개라 하나만 선택
    root = centers[0]

    def visit_function(u, v):
        # 방문했을때 수행할 함수.
        # C++ CalcRotation functor 대응
        idx = u*n + v
        m = pairwise_matches[idx]
        H = m.H.astype(np.float64)
        
        # K_from, K_to
        # Kf, Kt 는 각각 중심 이미지와 v 이미지의 카메라 행렬
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

        # 여기서 H 는 중심노드에서부터, 인접한 이웃 노드를 BFS order대로 방문하며 중심 이미지를 v 이미지 로 변환하는 호모그래피 행렬.
        # 모든 v 이미지를 중심이미지로 변환하는 Rotation 행렬을 계산
        # H = Kt^-1 * R * Kf        # 중심 이미지에서 v 이미지로의 변환 행렬
        # R = Kf * H * Kt^-1        # 중심 이미지에서 v 이미지로의 변환 행렬
        # R^-1 = Kf^-1 * H^-1 * Kt  # v 이미지에서 중심 이미지로의 변환 행렬
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
    이미지를 원통형으로 왜곡하는 함수
    
    Args:
        img (numpy.ndarray): 입력 이미지
        f_width (float): 가로 방향 초점 거리 (픽셀 단위)
        f_height (float): 세로 방향 초점 거리 (픽셀 단위)
        c_w (float): 이미지 중심 x 좌표
        c_h (float): 이미지 중심 y 좌표
        
    Returns:
        numpy.ndarray: 원통형으로 왜곡된 이미지
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