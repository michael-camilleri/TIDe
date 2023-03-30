"""
A set of wrappers around the VGG FCOS architectures.
"""
# ---- Standard Libraries ---- #
from mpctools.parallel import ProgressBar
from mpctools.extensions import cvext
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
import sys

# ---- FCOS ---- #
from fcos_core.config import cfg as default_cfg
from demo.mouse_predictor import COCODemo


class FCOSDetector:
    """
    Wrapper around FCOS Detector. This is based on the COCODemo object.

    Note that as per Coco, Images are expected in BGR format (as per OpenCV standard)
    """

    # Set of Key names to use.
    KEYS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")

    #  Super-Set Naming: (mouse/tunnel)
    SUPER = ["M", "T"]

    @staticmethod
    def __overlap_ratio(bbox: cvext.BoundingBox, polygon: Polygon):
        """
        Compute the Percentage (of bbox) area of overlap
        :param bbox:    Bounding Box detection
        :param polygon: Exclusion Zone
        :return:        Ratio of the bbox in the exclusion polygon
        """
        return Polygon(shell=bbox.corners).intersection(polygon).area / bbox.area()

    def __init__(
        self,
        mdl_pth,
        mdl_cfg,
        max_predictions=(3, 1),
        thresholds=(0.3, 0.3),
        device=None,
    ):
        """
        Initialises the Model

        :param mdl_pth:
        :param mdl_cfg:
        :param max_predictions: Will output at maximum these predictions for each of the classes.
        :param thresholds: Confidence Score thresholds for the Mouse/Tunnel respectively
        :param device:
        """
        # Set up Configuration
        self.Cfg = default_cfg.clone()
        self.Cfg.merge_from_file(mdl_cfg)
        self.Cfg.MODEL.WEIGHT = mdl_pth
        self.Cfg.TEST.DETECTIONS_PER_IMG = int(np.sum(max_predictions) * 2)
        if device is not None:
            self.Cfg.MODEL.DEVICE = device
        self.Cfg.freeze()
        self.max_predictions = max_predictions

        # Set up Model
        self.Mdl = COCODemo(self.Cfg, thresholds, min_image_size=720)

    def detect_in_frame(self, frame, exc_zones=(None, None), exc_ratios=(None, None)):
        """
        Detect mice & roll

        :param frame:  OpenCV Frame to detect within (BGR Format, as per OpenCV)
        :param exc_zones: Tuple of Polygons (shapely.Polygon) within which detections are
                          discarded. Note that if any one is not None, then the corresponding
                          exc_ratio must also not be None.
        :param exc_ratios: Tuple of Ratio of Area Overlap for discarding. Note that if any one is
                           not None, then the corresponding exc_zone must not be None.
        :return: Pandas Series, with a triple-index ([M/T]+ID+attribute), where attribute is:
                    * Det.BB: BB obtained by Detector
                    * Det.S: Score obtained by Detector
                 In addition, if exclusion_stats is True, then also return a Series with Number
                 of detections truncated by exclusion zones.
        """
        # ---- Prepare Storage ---- #
        a_detect = [
            [{"Det.BB": np.NaN, "Det.S": np.NaN,} for _ in range(self.max_predictions[i])]
            for i in range(len(self.SUPER))
        ]
        a_keys = [self.KEYS[: self.max_predictions[i]] for i in range(len(self.SUPER))]

        # ---- Get Candidates ---- #
        top_predict = self.Mdl.select_top_predictions(self.Mdl.compute_prediction(frame))

        # ---- Split into Mice/Tunnels and Store ---- #
        if len(top_predict) > 0:
            top_bboxes = np.asarray(
                [cvext.BoundingBox(tl=bb[:2], br=bb[2:]) for bb in top_predict.bbox]
            )
            top_labels = top_predict.get_field("labels").numpy()
            top_scores = top_predict.get_field("scores").numpy()
            # ++ Iterate over objects ++ #
            for obj in range(len(self.SUPER)):
                # Select the object/thresholds of interest
                o_labels = top_labels == (obj + 1)
                # Select the Scores/BBoxes corresponding to this
                if not o_labels.any():
                    continue
                o_scores = top_scores[o_labels]
                o_bboxes = top_bboxes[o_labels]
                # If there is a ratio selection, further select those who pass overlap ratio
                if exc_ratios[obj] is not None:
                    o_labels = np.asarray(
                        [self.__overlap_ratio(bb, exc_zones[obj]) < exc_ratios[obj] for bb in o_bboxes]
                    )
                    if not np.any(o_labels):
                        continue
                    o_scores = o_scores[o_labels]
                    o_bboxes = o_bboxes[o_labels]
                # Keep Top K
                o_len = min(np.sum(o_labels), self.max_predictions[obj])
                o_scores = o_scores[:o_len]
                o_bboxes = o_bboxes[:o_len]
                for m, b, s in zip(range(o_len), o_bboxes, o_scores):
                    a_detect[obj][m]["Det.BB"] = b
                    a_detect[obj][m]["Det.S"] = float(s)

        # ---- Build into Series and return ---- #
        dets = pd.concat(
            [
                pd.concat([pd.Series(det) for det in a_detect[i]], axis=0, keys=a_keys[i])
                for i in range(len(self.SUPER))
            ],
            axis=0,
            keys=self.SUPER,
        )
        return dets

    def detect_in_video(self, fg, within, exc_zones=(None, None), exc_ratios=(None, None),
                        show_progress=sys.stdout):
        """
        Detects the mice and tunnel rolls in a set of frames. This is a convenience wrapper
        around detect_objects which operates on a per-frame basis.

        :param fg:  Frame-Getter (supporting []-indexing) or VideoParser (supporting read()) object.
        :param within: The set of frames to look at (i.e. an iterable). For the VideoParser,
                       this will simply serve to index into the mse_ant dataframe, and the
                       VideoParser must have been initialised correctly (including taking care of
                       any frame offsets) - this also means that there cannot be any 'skipping'
                       of frames!
        :param exc_zones: Tuple of Shapely.Polygon instances (see detect_in_frame)
        :param exc_ratios: Tuple of Ratios (see detect_in_frame)
        :param show_progress: If specified, then shows a progress bar.
        :return: A DataFrame of tracked bounding boxes at each point, for each mouse/tunnel,
        indexed by Frame Number. The Columns (for each mouse/tunnel) are as follows:
            * Det.BB: BB obtained by Detector
            * Det.S: Score obtained by Detector
        """
        detections = []
        progress = ProgressBar(len(within), sink=show_progress).reset("Detecting")
        for frm in within:
            # ---- Extract Image ---- #
            if hasattr(fg, "__getitem__"):
                frame = fg[frm]
            else:
                ret, frame = fg.read()
                if not ret:
                    raise RuntimeError("Reached File end before all frames processed")
            # ---- Get Detections ---- #
            detections.append(self.detect_in_frame(frame, exc_zones, exc_ratios))
            # ---- Update Progress ---- #
            progress.update()

        detections = pd.concat(detections, axis=1, keys=within).T
        for i, o in enumerate(self.SUPER):
            for c in self.KEYS[: self.max_predictions[i]]:
                detections[(o, c, "Det.S")] = detections[(o, c, "Det.S")].astype(float)
        return detections
