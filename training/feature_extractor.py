import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

IMG_SIZE = 128

# LBP settings
LBP_P = 8
LBP_R = 1
LBP_BINS = LBP_P + 2


def fill_holes(mask):
    mask = ((mask > 0) * 255).astype(np.uint8)

    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, holes)

    return filled


def pick_best_component(mask, hsv):
    """
    Choose the best connected component instead of blindly taking the largest.
    Gives preference to:
    - larger area
    - higher saturation
    - more central position
    Penalizes components touching the image border.
    """
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return mask

    H, W = mask.shape
    S = hsv[:, :, 1]

    best_score = -1
    best_mask = np.zeros_like(mask, dtype=np.uint8)
    min_area = int(0.005 * H * W)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        comp = (labels == i)
        mean_sat = float(S[comp].mean()) if np.any(comp) else 0.0

        cx, cy = centroids[i]
        center_dist = np.sqrt((cx - W / 2) ** 2 + (cy - H / 2) ** 2) / np.sqrt((W / 2) ** 2 + (H / 2) ** 2)
        center_bonus = max(0.0, 1.0 - center_dist)

        touches_border = (x <= 1 or y <= 1 or (x + w) >= (W - 1) or (y + h) >= (H - 1))
        border_penalty = 0.75 if touches_border else 1.0

        score = area * (1.0 + mean_sat / 255.0) * (0.75 + 0.5 * center_bonus) * border_penalty

        if score > best_score:
            best_score = score
            best_mask[:] = 0
            best_mask[comp] = 255

    return best_mask


def border_fraction(mask):
    m = mask > 0
    total = np.count_nonzero(m)
    if total == 0:
        return 0.0

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    return np.count_nonzero(m & border) / total


def build_candidate_mask(image_bgr, use_center_prior=False):
    """
    Builds a leaf-like candidate mask before choosing the best component.
    """
    if image_bgr.shape[:2] != (IMG_SIZE, IMG_SIZE):
        image_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_bg_ratio = np.mean((S < 35) & (V > 170))

    green = (H >= 25) & (H <= 95) & (S > 40) & (V > 30)
    yellow = (H >= 12) & (H <= 40) & (S > 40) & (V > 50)

    if white_bg_ratio > 0.45:
        # white/paper background mode
        base = green | yellow | ((S > 45) & (V > 45) & (V < 245))
    else:
        # natural scene mode
        base = green | yellow

        if use_center_prior:
            yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
            cx, cy = IMG_SIZE / 2, IMG_SIZE / 2
            rx, ry = IMG_SIZE * 0.32, IMG_SIZE * 0.40
            center_prior = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
            base = base & center_prior

    mask = (base.astype(np.uint8) * 255)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = fill_holes(mask)

    return mask


def count_leaflike_components(image_bgr):
    """
    Count possible leaf-like regions using a stricter mask,
    before heavy closing merges nearby leaves together.
    """
    if image_bgr.shape[:2] != (IMG_SIZE, IMG_SIZE):
        image_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    green = (H >= 25) & (H <= 95) & (S > 45) & (V > 35)
    yellow = (H >= 12) & (H <= 40) & (S > 45) & (V > 50)

    raw = ((green | yellow).astype(np.uint8) * 255)

    # Use only a light opening, no strong closing
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k_open)

    binary = (raw > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = int(0.01 * IMG_SIZE * IMG_SIZE)

    # Glare, wet-surface reflections, and shadows can fall inside the
    # green/yellow hue range while looking visibly washed-out compared to
    # actual leaf tissue. Require decent saturation so these don't get
    # counted as leaf-like regions.
    min_saturation = 90

    candidate_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp_mask = (labels == i)
        mean_sat = float(S[comp_mask].mean())
        if mean_sat < min_saturation:
            continue

        candidate_areas.append(area)

    if not candidate_areas:
        return 0

    # A finger sliver or stray saturated speck can still pass the checks
    # above on its own while being tiny compared to the actual leaf. Only
    # count a region as a genuinely separate leaf if it's a substantial
    # fraction of the biggest qualifying region — two real leaves are
    # usually comparable in size, whereas incidental matches are not.
    largest = max(candidate_areas)
    significant_fraction = 0.25
    count = sum(1 for a in candidate_areas if a >= significant_fraction * largest)

    return count


def segment_leaf(image_bgr):
    """
    Final segmentation:
    - works better for white background
    - focuses on one central leaf in natural scenes
    - rejects obviously bad masks

    Returns (mask, leaf_ratio, rejected) where `rejected` is True whenever
    segmentation bailed out (empty mask, too small, too large, or hugging
    the border). This is a strong "this probably isn't a kangkong leaf"
    signal that gets fed into the feature vector for the unclassified class.
    """
    if image_bgr.shape[:2] != (IMG_SIZE, IMG_SIZE):
        image_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_bg_ratio = np.mean((S < 35) & (V > 170))

    mask = build_candidate_mask(
        image_bgr,
        use_center_prior=(white_bg_ratio <= 0.45)
    )

    if np.count_nonzero(mask) == 0:
        empty = np.zeros_like(mask, dtype=np.uint8)
        return empty, 0.0, True

    mask = pick_best_component(mask, hsv)

    # Add nearby brown lesion areas around the selected leaf
    brown = ((((H <= 20) | (H >= 165)) & (S > 45) & (V > 45) & (V < 190))).astype(np.uint8) * 255
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
    brown_near_leaf = cv2.bitwise_and(brown, dilated)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.bitwise_or(mask, brown_near_leaf)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = fill_holes(mask)

    leaf_ratio = np.count_nonzero(mask) / mask.size
    bf = border_fraction(mask)

    if leaf_ratio < 0.02 or leaf_ratio > 0.65 or bf > 0.08:
        empty = np.zeros_like(mask, dtype=np.uint8)
        return empty, leaf_ratio, True

    return mask, leaf_ratio, False


def masked_pixels(channel, mask):
    vals = channel[mask > 0]
    if len(vals) == 0:
        vals = channel.flatten()
    return vals


def color_hist_features(hsv, mask):
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def lbp_features(gray, mask):
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    vals = lbp[mask > 0]
    if len(vals) == 0:
        vals = lbp.flatten()

    hist, _ = np.histogram(vals, bins=np.arange(0, LBP_BINS + 1), range=(0, LBP_BINS))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-8)
    return hist


def glcm_features(gray, mask):
    masked_gray = gray.copy()
    masked_gray[mask == 0] = 0

    glcm = graycomatrix(
        masked_gray,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2],
        levels=256,
        symmetric=True,
        normed=True
    )

    props = []
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).flatten()
        props.extend(vals)

    return np.array(props, dtype=np.float32)


def edge_density_features(gray, mask):
    """
    Disease lesions tend to have sharp, high-contrast borders (spots),
    while overall discoloration/chlorosis tends to be a smooth gradient
    with far fewer hard edges. Edge density inside the leaf helps tell
    the two apart.
    """
    edges = cv2.Canny(gray, 60, 150)
    mask_pixels = np.count_nonzero(mask)
    if mask_pixels == 0:
        mask_pixels = mask.size
        edge_count = np.count_nonzero(edges)
    else:
        edge_count = np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask))

    return np.array([edge_count / mask_pixels], dtype=np.float32)


def lesion_shape_features(brown_mask, mask_pixels):
    """
    Characterizes the brown/necrotic regions found inside the leaf:
    - how many distinct spots there are
    - how big they are on average, relative to leaf size
    - how "spot-like" (compact/circular) vs. "diffuse/patchy" they are

    Disease (diseased) tends to show multiple small, roughly circular
    lesions. Simple discoloration (discolored) tends to show one or two
    large, irregular, low-circularity patches (or none at all). This is
    the main feature block aimed at separating those two classes.
    """
    binary = (brown_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if mask_pixels == 0:
        mask_pixels = brown_mask.size

    if len(contours) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    areas = []
    circularities = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 2:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        areas.append(area)
        circularities.append(min(circularity, 1.0))

    if len(areas) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    areas = np.array(areas, dtype=np.float32)
    circularities = np.array(circularities, dtype=np.float32)

    lesion_count = float(len(areas))
    mean_area_ratio = float(np.mean(areas) / mask_pixels)
    std_area_ratio = float(np.std(areas) / mask_pixels)
    mean_circularity = float(np.mean(circularities))
    # Many small spots vs. one big patch, independent of total lesion area.
    fragmentation = float(lesion_count / (np.sum(areas) / mask_pixels + 1e-8))

    return np.array(
        [lesion_count, mean_area_ratio, std_area_ratio, mean_circularity, fragmentation],
        dtype=np.float32
    )


def engineered_features(image_bgr, hsv, gray, mask, leaf_ratio, seg_rejected):
    h, s, v = cv2.split(hsv)

    mask_pixels = np.count_nonzero(mask)
    if mask_pixels == 0:
        mask_pixels = mask.size

    yellow_mask = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([35, 255, 255]))
    yellow_ratio = np.count_nonzero(cv2.bitwise_and(yellow_mask, mask)) / mask_pixels

    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
    green_ratio = np.count_nonzero(cv2.bitwise_and(green_mask, mask)) / mask_pixels

    brown1 = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([20, 255, 180]))
    brown2 = cv2.inRange(hsv, np.array([160, 30, 20]), np.array([179, 255, 180]))
    brown_mask = cv2.bitwise_or(brown1, brown2)
    brown_mask = cv2.bitwise_and(brown_mask, mask)

    lesion_ratio = np.count_nonzero(brown_mask) / mask_pixels

    # Dark/necrotic centers (true disease often has near-black spot centers;
    # plain discoloration/chlorosis usually doesn't).
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 55]))
    dark_mask = cv2.bitwise_and(dark_mask, mask)
    dark_ratio = np.count_nonzero(dark_mask) / mask_pixels

    h_vals = masked_pixels(h, mask)
    s_vals = masked_pixels(s, mask)
    v_vals = masked_pixels(v, mask)
    gray_vals = masked_pixels(gray, mask)

    mean_h, std_h = float(np.mean(h_vals)), float(np.std(h_vals))
    mean_s, std_s = float(np.mean(s_vals)), float(np.std(s_vals))
    mean_v = float(np.mean(v_vals))
    std_gray = float(np.std(gray_vals))

    black_on_yellow = dark_ratio * yellow_ratio

    # Shape analysis on the dark mask itself (not a stricter re-threshold —
    # that was missing real spots, as confirmed by debugging).
    black_spot_shape = lesion_shape_features(dark_mask, mask_pixels)
    has_black_spot = 1.0 if black_spot_shape[0] > 0 else 0.0  # lesion_count > 0

    base_features = np.array([
        yellow_ratio,
        green_ratio,
        lesion_ratio,
        dark_ratio,
        mean_h,
        std_h,
        mean_s,
        std_s,
        mean_v,
        std_gray,
        leaf_ratio,
        1.0 if seg_rejected else 0.0,
        black_on_yellow,
        has_black_spot,
    ], dtype=np.float32)

    lesion_shape = lesion_shape_features(brown_mask, mask_pixels)

    return np.concatenate([base_features, lesion_shape, black_spot_shape]).astype(np.float32)

def extract_features_from_bgr(image_bgr):
    image = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    mask, leaf_ratio, seg_rejected = segment_leaf(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    f_color = color_hist_features(hsv, mask)
    f_lbp = lbp_features(gray, mask)
    f_glcm = glcm_features(gray, mask)
    f_edge = edge_density_features(gray, mask)
    f_eng = engineered_features(image, hsv, gray, mask, leaf_ratio, seg_rejected)

    features = np.concatenate([f_color, f_lbp, f_glcm, f_edge, f_eng]).astype(np.float32)
    return features


def extract_features(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return extract_features_from_bgr(image)
