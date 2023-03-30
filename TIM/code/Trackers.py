# ---- Standard Libraries ---- #
from mpctools.extensions import cvext, npext, utils
from mpctools.parallel import ProgressBar
from argparse import Namespace
import pandas as pd
import numpy as np
import time as tm

# ---- Siam Mask Tracker ---- #
from experiments.siammask_sharp.custom import Custom
from utils import config_helper as lz_ch
from utils import load_helper as lz_lh
from tools import test as lz_sm

# ---- SORT Tracker ---- #
from sort import Sort


class SORTTracker:
    """
    Wrapper around SORT Tracker for my Data

    Currently only supports the SORT (not the DeepSORT) version, and only focuses on a single class.
    """
    @staticmethod
    def select_mice(detections, cutoff=0.4, max_det=5):
        """
        Convenience function to select only the Mouse Detections above a score threshold
        :param detections: Detections DataFrame
        :param cutoff: The Score Threshold
        :param max_det: Maximum number of detections per Frame
        :return:
        """
        detections = detections['M'].stack(0, dropna=False)
        detections = detections.loc[detections["Det.S"] > cutoff, "Det.BB"].unstack(1)
        return detections.rename_axis('Frame').iloc[:, :max_det]

    def __init__(self, max_hid, min_length, min_iou):
        """
        Initialises
        :param max_hid: (SORT Parameter) The maximum number of frames to allow a tracklet to be
                        hidden before it is considered dead.
        :param min_length: The minimum number of successive frames to consider a tracklet to be
                           viable. If 1 or less, then all are considered valid.
        :param min_iou: (SORT Parameter) The minimum IoU for assignment.
        """
        self.__max_hidden = max_hid
        self.__min_iou = min_iou
        self.__min_lifetime = max(min_length, 1)

    def track(self, detections, compute_iou=False, show_progress=None):
        """
        Track the Detections together

        :param detections: Pandas DataFrame of Detections, with columns indicating different
                            detections in each frame. This should contain bounding boxes for each
                            frame, and must exist for contiguous frames.
        :param compute_iou: If True, compute also the IoU between successive frames in a tracklet.
        :param show_progress: If not None, a sink to show a progress-bar
        :return: DataFrame with Tracklets. The rows are doubly-indexed, by sample and tracklet ID.
                 The columns are either 3 or 5:
                    * Trk.BB.Det : [Always] The BBox as detected by the Detector
                    * Trk.IoU.Det : [Optional] The IoU difference with the previous frame (for Det)
                    * Trk.BB.KFs : [Always] The BBox as smoothed by the Kalman Filter
                    * Trk.IoU.KFs : [Optional] The IoU difference with the previous frame (for KF)
                    * Det.ID : [Always] The original Detection ID (A-E) for evaluatin purposes
        """
        # ==== Ensure Indexing is Continuous ==== #
        detections = detections.sort_index()
        detections = detections.reindex(np.arange(detections.index[0], detections.index[-1] + 1))

        # ==== Prepare ==== #
        progress = ProgressBar(len(detections), sink=show_progress).reset("Tracking")
        sink = utils.NullableSink(show_progress)
        tracker = Sort(max_age=self.__max_hidden, min_hits=0, iou_threshold=self.__min_iou)
        tracks = []

        # ==== Iterate over Frames ==== #
        for frm, row in detections.iterrows():
            # ---- Track ---- #
            bboxes = []
            for n, bb in row.dropna().items():
                bboxes.append([*bb.extrema, n])
            trcks = tracker.update(bboxes)
            # ---- Format and Consolidate ---- #
            for trck in trcks:
                tracks.append(
                    pd.Series(
                        {
                            "Trk.BB.Det": cvext.BoundingBox(tl=trck[2][:2], br=trck[2][2:]),
                            "Trk.BB.KFs": cvext.BoundingBox(tl=trck[1][:2], br=trck[1][2:]),
                            "Trk.ID": trck[0],
                            "Det.ID": trck[3]
                        },
                        name=frm,
                    )
                )
            # ---- Update Progress ----- #
            progress.update()

        # ==== Consolidate ==== #
        # ---- Concatenate and Re-Index like Detections ---- #
        sink.write('Concatenating/Re-Indexing... ')
        s = tm.time()
        tracks = pd.concat(tracks, axis=1).T.rename_axis('Frame').set_index("Trk.ID", append=True)
        tracks = tracks.reindex(detections.index, level=0, copy=False)
        sink.write(f'[Done ({utils.show_time(tm.time() - s)})]\n')
        # ---- Remove Tracklets which do not pass Lifetime Threshold ---- #
        if self.__min_lifetime > 1:
            sink.write('Checking Lifetimes... ')
            s = tm.time()
            tracks = self.trim_lifetimes(tracks, self.__min_lifetime)
            sink.write(f'[Done ({utils.show_time(tm.time() - s)})]\n')
        # ---- Compute IoU Scores ---- #
        if compute_iou:
            sink.write('Computing IoU... ')
            s = tm.time()
            tracks = tracks.groupby('Trk.ID').apply(self.__compute_iou)
            sink.write(f'[Done ({utils.show_time(tm.time() - s)})]\n')

        # ==== Return ==== #
        return tracks

    @staticmethod
    def trim_lifetimes(tracklets, lifetime):
        """
        Thresholds Tracklets based on life-time
        :param tracklets: DataFrame of Tracklets
        :param lifetime: Life-time Threshold
        :return: Trimmed DataFrame
        """
        def longest_run(grp):
            return npext.run_lengths(npext.contiguous(grp.index.get_level_values(0))).max()

        max_lifetime = tracklets.groupby('Trk.ID').apply(longest_run)
        tracklets = tracklets.drop(index=max_lifetime[max_lifetime < lifetime].index, level=1)
        trk_map = {k: i for i, k in enumerate(tracklets.index.unique(1).sort_values())}
        return tracklets.rename(index=trk_map, level=1)

    @staticmethod
    def __compute_iou(grp):
        grp = grp.sort_index(level=1) # Ensure that Sorted
        for col in ('Det', 'KFs'):
            ccat = pd.concat([grp[f'Trk.BB.{col}'], grp[f'Trk.BB.{col}'].shift()], axis=1)
            grp[f'Trk.IoU.{col}'] = ccat.apply(lambda row: row[0].iou(row[1]) if row.notna().all() else np.NaN, axis=1)
        return grp


class SiamMaskTracker:
    """
    Wrapper around Li's Siamese Tracker
    """

    MASK_ENABLE = False

    def __init__(self, mdl_pth, mdl_cfg, device):
        """
        Initialises the Tracker

        :param mdl_pth: A Model State file (pytorch format)
        :param mdl_cfg: Model Configuration file as specified by SiamMask
        :param device: cpu or cuda
        """
        self.Cfg = lz_ch.load_config(Namespace(config=mdl_cfg, arch="Custom"))
        self.Mdl = lz_lh.load_pretrain(Custom(anchors=self.Cfg["anchors"]), mdl_pth)
        self.Mdl.eval().to(device)
        self.Dvc = device

    def track_mouse(self, fg, within, bb, show_progress=None):
        """
        Tracks a mouse within a range of frames

        :param fg:  An object which can retrieve individual frames by index in []-style. This
                    allows separating the exact dynamics of how images are retrieved from the
                    tracking.
        :param within: 3-Tuple showing the (0) start of the track, (1) the TIP-frame and (2) the end
                    of the track. This must be in absolute frame numbers.
        :param bb: The Bounding box at the TIP (cvext.BoundingBox)
        :param show_progress: If specified, then shows a progress bar.
        :return: A DataFrame of tracked bounding boxes at each point, for the specific mouse,
        indexed by Frame Number
            * RTrack.BB: BB obtained by reverse tracking
            * RTrack.S:  Score given by Siamese Tracker
            * Init.BB: (copy of bb)
            * Init.S: Score given for initialisation (since this provided, it is 1.0)
            * FTrack.BB: BB obtained by forward tracking
            * FTrack.S: Score given by Siamese Tracker
        """
        # Initialise track with Init
        tracks = {"Frame": [within[1]], "Trk.BB": [bb], "Trk.S": [1.0], "Trk.Dir": ['I']}

        # Backwards Pass
        state = lz_sm.siamese_init(
            fg[within[1]], bb.center, bb.size, self.Mdl, self.Cfg["hp"], self.Dvc
        )
        progress = ProgressBar(within[1] - within[0], sink=show_progress).reset("Back-Tracking   ")
        for frm in reversed(range(within[0], within[1])):  # We want within[0] but not within[1]
            state = lz_sm.siamese_track(state, fg[frm], self.MASK_ENABLE, True, self.Dvc)
            tracks["Frame"].append(frm)
            tracks["Trk.BB"].append(cvext.BoundingBox(c=state["target_pos"], sz=state["target_sz"]))
            tracks["Trk.S"].append(state["score"])
            tracks["Trk.Dir"].append('R')
            progress.update()

        # Forward Pass
        state = lz_sm.siamese_init(
            fg[within[1]], bb.center, bb.size, self.Mdl, self.Cfg["hp"], self.Dvc
        )
        progress = ProgressBar(within[2] - within[1], sink=show_progress).reset("Forward-Tracking")
        for frm in range(within[1] + 1, within[2] + 1):  # We do not want within[1] but within[2]
            state = lz_sm.siamese_track(state, fg[frm], self.MASK_ENABLE, True, self.Dvc)
            tracks["Frame"].append(frm)
            tracks["Trk.BB"].append(cvext.BoundingBox(c=state["target_pos"], sz=state["target_sz"]))
            tracks["Trk.S"].append(state["score"])
            tracks["Trk.Dir"].append('F')
            progress.update()

        # Just return as DataFrame
        return pd.DataFrame(data=tracks).set_index("Frame").sort_index()

