"""Turn a detection box into a front-on sign crop.

1. Fit the sign's outline quad inside the (padded) box. If none is found, use the box itself.
2. Recover the sign's true aspect ratio and plane normal from the quad and K
   (Zhang & He, "Whiteboard scanning and image enhancement", 2007). This works because
   the image is undistorted, so a pinhole model holds.
3. Warp the quad to a rectangle of that aspect.

info (all plain Python types, so it can go straight into reads.jsonl):
  theta       bearing of the sign centre in the camera, degrees, + = right
  frontality  |cos| of the angle between the sign normal and the viewing ray (1 = head-on)
  aspect      true width/height of the sign
  height_px   sign height in the image, i.e. how much resolution we have
  sharpness   variance of the Laplacian on a fixed-height copy of the crop
  truncated   box touches the image border (part of the sign is probably cut off)
  quad_found  whether step 1 found an outline"""
import cv2
import numpy as np


def order_corners(pts):
    """tl, tr, br, bl. Sorted by angle around the centroid, so it survives rotation."""
    pts = np.asarray(pts, float).reshape(4, 2)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # clockwise on screen
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)


def _box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _refine(quad, contour):
    """Sub-pixel corners: fit a line to the contour points along each side, intersect them.
    A 1-2px corner error on a small oblique sign visibly skews aspect and frontality."""
    pts = contour.reshape(-1, 2).astype(float)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            return quad
        t = (pts - a) @ d / L ** 2
        dist = np.abs((pts - a) @ np.array([-d[1], d[0]]) / L)
        sel = pts[(t > 0.1) & (t < 0.9) & (dist < max(2.0, 0.03 * L))]
        if len(sel) < 5:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append(np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0]))
    out = []
    for i in range(4):
        p = np.cross(lines[i - 1], lines[i])
        if abs(p[2]) < 1e-9:
            return quad
        out.append(p[:2] / p[2])
    out = np.array(out)
    return out if np.abs(out - quad).max() < 0.1 * np.linalg.norm(quad[2] - quad[0]) else quad


def find_quad(image, box, pad=0.15, min_iou=0.6):
    """Largest convex 4-gon in the padded box whose bounding box agrees with the detection."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    X2, Y2 = min(w, int(x2 + pad * bw) + 1), min(h, int(y2 + pad * bh) + 1)
    roi = image[Y1:Y2, X1:X2]
    if roi.size == 0 or min(roi.shape[:2]) < 12:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8))
    best, best_c, best_iou = None, None, min_iou
    for binimg in (otsu, 255 - otsu, edges):
        cnts, _ = cv2.findContours(binimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.contourArea(c) < 0.4 * bw * bh:
                continue
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
            if len(approx) != 4:
                continue
            q = approx.reshape(4, 2).astype(float) + (X1, Y1)
            iou = _box_iou((*q.min(0), *q.max(0)), box)
            if iou > best_iou:
                best, best_c, best_iou = q, c + (X1, Y1), iou
    return None if best is None else _refine(order_corners(best), best_c)


def plane_geometry(quad, K):
    """(aspect w/h, unit normal facing the camera) of the 3D rectangle seen as `quad`, or None."""
    tl, tr, br, bl = [np.array([p[0], p[1], 1.0]) for p in quad]
    m1, m2, m3, m4 = tl, tr, bl, br
    d2, d3 = np.dot(np.cross(m2, m4), m3), np.dot(np.cross(m3, m4), m2)
    if abs(d2) < 1e-9 or abs(d3) < 1e-9:
        return None
    k2 = np.dot(np.cross(m1, m4), m3) / d2   # depth ratios: must be > 0 for a real rectangle
    k3 = np.dot(np.cross(m1, m4), m2) / d3
    if not (k2 > 0 and k3 > 0):
        return None
    Ki = np.linalg.inv(K)
    u = Ki @ (k2 * m2 - m1)   # 3D top edge, up to a common scale
    v = Ki @ (k3 * m3 - m1)   # 3D left edge, same scale
    aspect = np.linalg.norm(u) / np.linalg.norm(v)
    if not (0.1 < aspect < 10) or not np.isfinite(aspect):
        return None
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    if np.dot(n, Ki @ tl) > 0:   # point the normal back at the camera
        n = -n
    return float(aspect), n


def rectify(image, box, K, max_h=400, margin=0.06):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    box_quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
    quad = find_quad(image, (x1, y1, x2, y2))
    geom = plane_geometry(quad, K) if quad is not None else None
    if geom is None:
        quad = box_quad
        geom = plane_geometry(quad, K)   # an axis-aligned box means a fronto-parallel plane
    if geom is None:                     # extreme box shape (edge-on): just keep its pixel aspect
        geom = (float(np.clip((x2 - x1) / max(1.0, y2 - y1), 0.1, 10)), np.array([0.0, 0.0, -1.0]))
    found = quad is not box_quad
    aspect, n = geom

    tl, tr, br, bl = quad
    side = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    out_h = int(np.clip(side, 16, max_h))
    out_w = int(np.clip(out_h * aspect, 16, 4 * max_h))
    m = int(round(margin * out_h))   # keep a little context: arrows often sit at the edge
    dst = np.array([[m, m], [m + out_w, m], [m + out_w, m + out_h], [m, m + out_h]], np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    crop = cv2.warpPerspective(image, H, (out_w + 2 * m, out_h + 2 * m), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    c = quad.mean(0)
    ray = np.linalg.inv(K) @ np.array([c[0], c[1], 1.0])
    frontality = abs(float(np.dot(n, ray / np.linalg.norm(ray))))
    theta = float(np.degrees(np.arctan2(c[0] - K[0, 2], K[0, 0])))
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (max(8, int(96 * g.shape[1] / g.shape[0])), 96), interpolation=cv2.INTER_AREA)
    info = dict(
        theta=theta,
        frontality=frontality,
        aspect=aspect,
        height_px=float(0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))),
        sharpness=float(cv2.Laplacian(g, cv2.CV_64F).var()),
        truncated=bool(x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3),
        quad_found=bool(found),
        quad=[[round(float(a), 1), round(float(b), 1)] for a, b in quad],
    )
    return crop, info
