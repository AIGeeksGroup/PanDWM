"""Shared constants and helpers of the controllability evaluation (2026-10-21).

Protocol (frozen 2026-10-21): experiments/PanDWM-baselines-20261019/CONTROLLABILITY_PLAN_20261021.md.
Images: the six cameras at 256x448 (cam_band_feats grid), mask = onebev_inverse_own with native rows 0..51 removed
(the paper's camera-space protocol, feats_val150_cam_onebev_inverse_own_top52). Camera axis = cam_band_feats.CAM_ORDER.
Box coordinates: 256x448 edge coordinates (pixel j covers [j, j + 1)), as torchvision boxes.
"""
import numpy as np

H, W = 256, 448                 # camera view size of the protocol
NW, NH = 1600, 900              # native nuScenes camera size
FIRST, LAST = 3, 34             # scored clip frames
NFR = LAST - FIRST + 1          # 32
TOPROWS = 52                    # MagicDrive-V2 padding rows (native), masked for every arm
NEAR = 0.5                      # m: camera-depth plane for "in front of the camera" and hull clipping
MIN_H = 16.0                    # px at 256 rows: GT height rule and the detection size range
VEH_RANGE, PED_RANGE = 50.0, 40.0
VIS_MIN = 3                     # nuScenes visibility token >= 3 (more than 60% visible)
VMASK_MIN = 0.5
OCC_MAX = 0.5
ROAD_RANGE = 50.0               # m, horizontal distance from the camera of the ground hit
NEAR_ROAD = 20.0                # m, near / far split of Road IoU (supplement)
HULL_RANGE = 60.0               # m, boxes whose projected hull is removed from the road region
RASTER_RES = 0.05               # m, drivable-area raster
RASTER_MARGIN = 60.0            # m, around the clip's camera positions
IOU_T, IOA_T, SCORE_T = 0.5, 0.5, 0.5
DET_MIN_SCORE, MAX_DETS = 0.05, 100
VEH, PED = 0, 1
CLASS_NAMES = ("vehicle", "pedestrian")
REQ, IGN = 0, 1

VEH_NAMES = ("car", "truck", "bus")
PED_NAMES = ("pedestrian",)
# COCO category ids as used by torchvision (91-index): person 1, car 3, bus 6, truck 8
COCO_CLASS = {1: PED, 3: VEH, 6: VEH, 8: VEH}
COCO_NAME_TO_ID = {"person": 1, "car": 3, "bus": 6, "truck": 8}


def veh_ignore_name(n):
    """Annotated objects a COCO detector calls car / truck but that are not vehicle GT."""
    return n in ("trailer", "construction_vehicle") or n.startswith("vehicle.")


def ped_ignore_name(n):
    """Annotated objects whose people a COCO detector calls person but that are not pedestrian GT."""
    return (n in ("bicycle", "motorcycle", "car", "truck", "bus", "trailer", "construction_vehicle")
            or n.startswith("vehicle.") or n.startswith("human.pedestrian."))


# cuboid edges and corner template of dwm_pano.datasets.pano_via_stitch (mmdet3d corner order); the pkl stores
# (w, l, h) with a counter-clockwise yaw (checked 2026-10-21: yaw - velocity heading = +-90 deg for moving cars)
EDGES = ((0, 3), (3, 7), (7, 4), (4, 0), (1, 2), (2, 6), (6, 5), (5, 1), (0, 1), (3, 2), (7, 6), (4, 5))
TEMPLATE = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
                     [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]])

POP = np.array([bin(i).count("1") for i in range(256)], np.int64)


def eval_mask(geometry, c2e, K, size):
    """The protocol mask [6, 256, 448] bool of a rig (cam_band_feats.LegacyInverseOwnGeometry and top 52 rows)."""
    geo = geometry(c2e, K, size)
    top = ((np.arange(H) + 0.5) * NH / H < TOPROWS)[None, :, None]
    return geo, geo["valid"] & ~top


def pack(b):
    """bool [..., 256, 448] -> uint8 [..., 14336]."""
    b = np.asarray(b, bool)
    return np.packbits(b.reshape(b.shape[:-2] + (H * W,)), axis=-1)


def unpack(p):
    p = np.asarray(p, np.uint8)
    return np.unpackbits(p, axis=-1).reshape(p.shape[:-1] + (H, W)).astype(bool)


def popcount(p, axis=-1):
    return POP[np.asarray(p, np.uint8)].sum(axis)


def box_iou(d, g, crowd=None):
    """IoU [nd, ng] of xyxy boxes; columns with crowd=True use intersection / detection area (COCO iscrowd)."""
    d = np.asarray(d, np.float64).reshape(-1, 4)
    g = np.asarray(g, np.float64).reshape(-1, 4)
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    ix = np.clip(np.minimum(d[:, None, 2], g[None, :, 2]) - np.maximum(d[:, None, 0], g[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(d[:, None, 3], g[None, :, 3]) - np.maximum(d[:, None, 1], g[None, :, 1]), 0, None)
    inter = ix * iy
    ad = ((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]))[:, None]
    ag = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))[None, :]
    union = ad + ag - inter
    if crowd is not None:
        union = np.where(np.asarray(crowd, bool)[None, :], ad, union)
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes, scores, thr):
    """Greedy NMS; returns kept indices in descending score order (stable for ties)."""
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    keep = []
    boxes = np.asarray(boxes, np.float64)
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        iou = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][iou <= thr]
    return np.array(keep, np.int64)
