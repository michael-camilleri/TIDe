"""
Implements Identification Scripts
"""
from scipy.spatial.distance import cdist, pdist, squareform
from mpctools.extensions import utils, npext, cvext
from Tools.Parsers import RFIDParser, VIAParser
from multiprocessing import Pool
from bidict import bidict
import itertools as it
import pandas as pd
import numpy as np
import time as tm
import mip
import os


class CentroidWeights:
    """
    Class to implement Centroid Based Distance weights.

    This class implements a weight metric based on distance between RFID and centroids in
    euclidean space. Inadmissability is based on a threshold on the same distance (returned as
    NINF). The values are returned as 1 - distance/max_distance.
    """

    @property
    def models_hidden(self):
        return False

    def __init__(self, ant_model, max_dist=500):
        """
        Initialises the Object

        :param ant_model: A Mapping from Antenna Index to Centroid (Pandas Series)
        :param max_dist: Maximum Distance to use: this also serves as the normaliser.
        """
        self.__model = ant_model
        self.__mdist = max_dist

    def weights(self, bbs, pos):
        """
        Computes the Weights for the bounding boxes and the rfid antennas

        :param bbs: Array of size [nB], representing detected BBoxes in frame
        :param pos: Array of size [nO], representing objects present in frame
        :return: Weight Matrix (negative distance). Invalid Assignments (cutoff) are NINF
        """
        # Compute Distance
        dist = cdist(np.vstack([b.center for b in bbs]), np.vstack(self.__model.loc[pos]), "euclidean")
        # Transform to Weight and return
        dist = 1 - dist / self.__mdist
        dist[dist < 0] = np.NINF
        return dist


class AntennaConditionalWeights:
    """
    Class to implement Antenna-Conditional Distributions

    This Class defines the weights as an antenna-conditional multivariate distribution on the
    centroid location and size (width/height) of the bounding box. Inadmissability is based on
    a threshold of log-probability (typically the log-probability for an outlier model)
    """

    @property
    def models_hidden(self):
        return False

    def __init__(self, bb_model, min_weight=None):
        """
        Initialises the Object

        :param model: The conditional probability model. Must be a series, indexed by antenna (1
            through 18), the elements of which implement a logpdf function.
        :param min_weight: Minimum Weight (log-likelihood).
        """
        self.__models = bb_model
        self.__min_weight = utils.default(min_weight, np.NINF)

    def weights(self, bbs, pos):
        """
        Computes the Weights for the bounding boxes and the rfid antennas

        :param bbs: Array of size [nB], representing detected BBoxes in frame
        :param pos: Array of size [nO], representing objects present in frame
        :return: Weight Matrix (log probability). Invalid Assignments (cutoff) are NINF
        """
        weights = np.zeros([len(bbs), len(pos)], dtype=float)
        for bi, bv in enumerate(bbs):
            for pi, pv in enumerate(pos):
                weights[bi, pi] = self.__models[pv].logpdf(bv.cs)
        weights[weights < self.__min_weight] = np.NINF
        return weights


class GenerativeWeights:
    """
    This implements the complete probabilistic observation model as defined in our BMVC Paper.

    Specifically, this computes the probability of RFID tag (or outlier model) having generated the
    bounding box (or hidden box). Note that in this case, there is no notion of a cutoff: this
    will be implicitly included due to the outlier model.

    Note the following Assumptions:
        1. A hidden bounding box is represented by a vector of 0's (relevant for the
            distribution under the hidden visibility)
        2. There is no further assumption about the form of the distribution of BB parameters
            under the hidden visibility flag.
        3. The only inadmissable assignment is between Outlier Model and Hidden Tracklet

    @TODO: Current implementation is extendible but potentially inefficient: consider removing the
        need for the hidden visibility model.
    """

    @property
    def models_hidden(self):
        return True

    def __init__(self, vis_model, bb_model, outlier_model, vis_smooth=1e-8):
        """
        Initialises the Weighting

        :param vis_model: The model used to predict the visibility flag (under the inlier
                hypothesis). The object should expose a predict_proba() method which accepts a
                10-long feature vector (Position Index, followed by 9-pt Occupancy Matrix).
        :param bb_model: The model used to predict the bounding box parameters (under the inlier
                hypothesis). This should be a series, doubly indexed by Visibility
                (VIAParser.[VIS_CLEAR|VIS_TRUNC|VIS_HID]) and Position Index (1...18). Each
                element should expose a pdf() method which accepts a 4-D vector.
        :param outlier_model: The model used to predict the bounding box parameters under the
                outlier hypothesis: should expose a logpdf() method which accepts a 4-D vector.
        :param vis_smooth: This is a smoothing value to add to the probabilities of the generated
                output, ensuring that there is never a 0-probability.
        """
        self._vis_model = vis_model
        self._bb_models = bb_model
        self._out_model = outlier_model
        self._alpha = vis_smooth

    def weights(self, bbs, pos):
        """
        Computes the assignment weights between bbs and rfid.

        :param bbs: Array of size [nB - 1], representing detected BBoxes in frame
        :param pos: Array of size [nO - 1], representing objects present in frame
        :return: The nB x nO weight matrix, in the order of bbs/pos, except for the last index in
                 each dimension, which refers to the Hidden and Outlier models respectively.
        """
        # Some Sizes
        nB, nO = len(bbs) + 1, len(pos) + 1

        # Generate Visibility Flag for each Object under Neighbourhood configuration
        _features = np.zeros([nO - 1, 10])
        for pi, pv in enumerate(pos):
            _features[pi, 0] = pv
            _features[pi, 1:] = RFIDParser.occupancy(pv, np.delete(pos, pi)).ravel()
        v = npext.sum_to_one(self._vis_model.predict_proba(_features) + self._alpha, axis=1)

        # Now Generate weights
        weight = np.empty([nB, nO], dtype=float)
        # --- Start with the Physical Objects and Detected BBs
        for bi, bb in enumerate(bbs):
            for pi, pv in enumerate(pos):
                weight[bi, pi] = np.log(
                    v[pi, :] @ self._bb_models.loc[(slice(None), pv)].apply(lambda x: x.pdf(bb.cs))
                )
        # --- Now Physical Objects and Hidden BB
        for pi, pv in enumerate(pos):
            weight[-1, pi] = np.log(
                v[pi, -1] * self._bb_models.loc[(VIAParser.VIS_HID, pv)].pdf([0, 0, 0, 0])
            )
        # --- Now Outlier Model and all BBs
        for bi, bb in enumerate(bbs):
            weight[bi, -1] = self._out_model.logpdf(bb.cs)
        # --- Finally Hidden against Outlier (Inadmissable, but set to 0 just in case)
        weight[-1, -1] = 0

        # Return Weight Matrix
        return weight


class CoveringIdentifier:
    """
    This Class implements the Identification as a formulation of the Covering Problem. The
    details are as indicated in the BMVC Paper.

    Comments on output:
      1. The output is typically indexed by Frame and Tracklet ID (unless by_object is selected).
      2. All Tracklets are given a value or None.
    """

    @staticmethod
    def index_by_object(tracks, id_col, idx):
        """
        Wrapper to convert a flat representation into the object-centric version (typically,
        this is easier for visualisation)

        :param tracks:
        :return:
        """
        tracks = tracks.dropna(axis=0, subset=[id_col]).set_index(id_col, append=True)
        tracks = tracks.reset_index(level=1).unstack(-1).reorder_levels((1, 0), axis=1)
        return tracks.sort_index(axis=1).reindex(idx)

    def __init__(
        self,
        distance,
        max_time=None,
        bb_col="Trk.BB.Det",
        id_col="Obj.ID",
        by_object=True,
        n_jobs=8,
        check_binary=False,
    ):
        """
        Initialises the Identifier

        :param distance: The distance method to use: this must be an object which implements a
                         weights() method, that takes two dataframes.
        :param max_time: Maximum time to allow for solving the problem. If None, do not bound.
        :param bb_col: The column in the Tracklets dataframe containing bounding boxes
        :param id_col: The column name to use for IDs
        :param by_object: If True, the output will be with different objects along the columns (
            as a column index). Otherwise, the Object is just a column rather than a column name.
        :param n_jobs: Number of jobs to run in parallel
        :param check_tu: If True, then ensure that LP is integral.
        """
        self.__max_time = utils.default(max_time, np.Inf)
        self.__dist_obj = distance
        self.__jobs = n_jobs
        self.__by_obj = by_object
        self.__bb_col = bb_col
        self.__id_col = id_col
        self.__binary = [] if check_binary else None
        self.__result = None

    def identify(self, tracks, objects, affine=None, store_w=None, sink=None):
        """
        Identify the Tracklets according to the Positions of Objects

        :param tracks: The Pandas DataFrame containing Tracklets. The index consists of two levels:
            Level 0 represents the Frames (contiguous, size F), and Level 1 the Tracklet ID. The
            main column should be named as per bb_col, and should contain BoundingBox objects.
            **May be modified in place!**
        :param objects: The Pandas Series containing Object-identifiers. The index is again two
            levels: L0 is the Frame number (again contiguous, size F) and L1 the Object ID. Each
            cell should be a feature (typically position index).
        :param affine: If not None, then transform the Bounding Boxes into some frame for the
            purpose of computing probabilities.
        :param store_w: Since computation of the weights is expensive, I provide the option to
            store it and retrieve it later. If this is a path (not None), two things happen:
                1. If the file exists, it is loaded from disk: otherwise (the first time round)
                   it is generated as usual.
                2. The generated W is stored to file.
        :param sink: Debugging
        :return: Assignment of tracklets to objects, as a DataFrame. Format depends on by_object
            setting. If the assignment failed, None is returned.
        """
        sink = utils.NullableSink(sink)
        if type(tracks) is pd.Series:  # Guard against passing series
            tracks = tracks.to_frame(self.__bb_col)

        # ==== Initial Setup (including setting up Index Maps) ==== #
        sink.write("Preparing ... ")
        s = tm.time()
        # ---- Indexing ---- #
        t_ids = bidict({t: i for i, t in enumerate(tracks.index.unique(1).sort_values())})
        o_ids = bidict({o: i for i, o in enumerate(objects.index.unique(1).sort_values())})
        trk_ = tracks.rename(index=t_ids, level=1, copy=True)
        obj_ = objects.rename(index=o_ids, level=1)
        # ---- Store/Initialise Constants ---- #
        sI, sJ = len(t_ids), len(o_ids)
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # ==== Affine Transforms ==== #
        sink.write("Applying Transform ... ")
        s = tm.time()
        if affine is not None:
            trk_[self.__bb_col] = trk_[self.__bb_col].apply(lambda bb: bb.transform(affine))
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # ==== Prepare L Matrix ==== #
        sink.write("Preparing L ... ")
        s = tm.time()
        L = []
        prev = None
        tidx = sI - 1
        f2t = {}  # Mapping from Frame to Time-Point
        for frm, trks in trk_.reset_index(level=1)["Trk.ID"].groupby("Frame"):
            if not np.array_equal(trks.values, prev):
                tidx += 1
                prev = trks.values
                L.append(np.append(trks.values, tidx))
            f2t[frm] = tidx
        sT = len(L)
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # ==== Prepare W Matrix ==== #
        sink.write("Preparing W ... ")
        if store_w is not None and os.path.isfile(store_w):
            sink.write("(Loading) ... ")
            W = np.load(store_w, allow_pickle=True)
        else:
            sink.write("(Computing) ... ")
            trk_obj = pd.concat([trk_[self.__bb_col], obj_], keys=["T", "O"])
            trk_obj = trk_obj.reorder_levels((1, 0, 2)).sort_index()
            if self.__jobs == 1:
                W = self._compute_weights((trk_obj, sI, sJ, sT, f2t))
            else:
                with Pool(self.__jobs) as p:
                    idcs = np.arange(trk_obj.index[0][0], trk_obj.index[-1][0] + 1)
                    data = [
                        (trk_obj.loc[l[0] : l[-1]], sI, sJ, sT, f2t)
                        for l in filter(len, np.array_split(idcs, self.__jobs))
                    ]
                    W = np.stack(p.map(self._compute_weights, data)).sum(axis=0)
        if store_w is not None:
            np.save(store_w, W, allow_pickle=True)
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # ==== Setting Up Problem ==== #
        sink.write("Setting Up ILP ... ")
        s = tm.time()
        mdl = mip.Model(name="Identifier", sense=mip.MAXIMIZE)
        #  => Objective
        a = [
            [mdl.add_var(var_type=mip.BINARY) for _ in range(sJ + 1)]
            for __ in range(sI + sT)
        ]
        mdl.objective = mip.maximize(
            mip.xsum(W[i, j] * a[i][j] for i in range(sI + sT) for j in range(sJ + 1))
        )
        #  => Constraint 1: Visible Tracklets assigned to one Mouse or Outlier Model
        for i in range(sI):
            mdl += mip.xsum(a[i][j] for j in range(sJ + 1)) == 1
        #  => Constraint 2: Mouse has only one tracklet at any point in time
        for (t, j) in it.product(range(sT), range(sJ)):
            mdl += mip.xsum(a[l][j] for l in L[t]) == 1
        #  => Constraint 3: Inadmissable Assignments
        for i in range(sI, sI + sT):
            mdl += a[i][sJ] == 0
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # --- Solve Cover --- #
        sink.write("Solving Cover ... ")
        s = tm.time()
        self.__result = mdl.optimize(max_seconds=self.__max_time, relax=False)
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # ==== Format and Return ==== #
        sink.write("Formatting Solution ... ")
        s = tm.time()
        if self.__result in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            tracks[self.__id_col] = None
            for (i, j) in it.product(range(sI), range(sJ)):
                a_ij = abs(a[i][j].x)
                if self.__binary is not None and not (np.isclose(a_ij, 1) or (np.isclose(a_ij, 0))):
                    self.__binary.append((i, j, a_ij))
                if abs(a[i][j].x) > 0.5:
                    tracks.loc[(slice(None), t_ids.inv[i]), self.__id_col] = o_ids.inv[j]
            if self.__by_obj:
                tracks = self.index_by_object(tracks, self.__id_col, objects.index.unique(0))
        else:
            tracks = None
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        return tracks

    @property
    def binary(self):
        return self.__binary

    @property
    def status(self):
        return self.__result

    def _compute_weights(self, data):
        """
        Wrapper for Generating the Weight Matrix

        :param trk_grp: A list of grouped frames.
        :return: Weight Matrix
        """
        trk_grp, sI, sJ, sT, f2t = data
        # Pre-Allocate Weight Matrix
        weight = np.zeros((sI + sT, sJ + 1), dtype=float)
        # Build up
        for fno, fdata in trk_grp.groupby(level=0):
            fdata = fdata.reset_index(0, drop=True)
            if "T" in fdata and "O" in fdata:  # No Assignment can be made otherwise
                bbs, obj = fdata["T"], fdata["O"]
                f_w = self.__dist_obj.weights(bbs.values, obj.values)  # Compute Weights
                weight[np.ix_(bbs.index, obj.index)] += f_w[:-1, :-1]  # Normal Assignments
                weight[bbs.index, -1] += f_w[:-1, -1]  # BBoxes to Outlier Model
                weight[f2t[fno], obj.index] += f_w[-1, :-1]  # Hidden Trk to Objects
                weight[f2t[fno], -1] += f_w[-1, -1]  # Hidden Trk to Outlier Model
        # Return Weight-Matrix
        return weight


class StaticIdentifier:
    """
    This class implements a non-temporal identification scheme.
    """

    def __init__(self, distance, bb_col="Det.BB", id_col="Obj.ID", n_jobs=10):
        """
        Initialises the Identifier

        :param distance: The distance method to use: this must be an object which implements a
                         weights() method, that takes two dataframes. This identifier can support
                         both weight models that consider the Hidden/Outlier weights or not.
        :param bb_col: The column in the Tracklets dataframe containing bounding boxes
        :param id_col: The column name to use for IDs
        :param n_jobs: Number of jobs to run in parallel.
        """
        self.__dist_obj = distance
        self.__bb_col = bb_col
        self.__id_col = id_col
        self.__jobs = n_jobs

    def identify(self, detections, objects, affine=None, sink=None):
        """
        Identify the Detections according to the Positions of Objects

        :param detections: DataFrame containing BBoxes from different detections. The index consists
                of two levels: Level 0 represents the Frames (contiguous, size F), and Level 1
                the Detection ID. The main column should be named as per bb_col, and should contain
                BoundingBox objects.
        :param objects: The Pandas Series containing Object-identifiers. The index is again two
                levels: L0 is the Frame number (again contiguous, size F) and L1 the Object ID. Each
                cell should be a feature (typically position index).
        :param affine: If not None, then transform the Bounding Boxes into some frame for the
                purpose of computing probabilities.
        :param sink: Debugging
        :return: Assignment of detections to objects, as a DataFrame. The index is again Frames
                and Detection IDs: one of the columns is the Object Identity assigned to each (or
                None).
        """
        sink = utils.NullableSink(sink)
        if type(detections) is pd.Series:  # Guard against passing series
            detections = detections.to_frame(self.__bb_col)
        det_ = detections.copy(deep=True)

        # ==== Affine Transforms ==== #
        sink.write("Applying Transform ... ")
        s = tm.time()
        if affine is not None:
            det_[self.__bb_col] = det_[self.__bb_col].apply(lambda bb: bb.transform(affine))
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # ==== Solve Assignment ==== #
        sink.write("Assigning Detections ... ")
        s = tm.time()
        if self.__jobs == 1:
            det_ = self._assign_detections((det_, objects))
        else:
            with Pool(self.__jobs) as p:
                idcs = np.arange(det_.index[0][0], det_.index[-1][0] + 1)
                data = [
                    (det_.loc[l[0] : l[-1]], objects.loc[l[0] : l[-1]])
                    for l in filter(len, np.array_split(idcs, self.__jobs))
                ]
                det_ = pd.concat(p.map(self._assign_detections, data))
        sink.write(f"[Done ({utils.show_time(tm.time() - s)})]\n")

        # Return
        return detections.join(det_)

    def _assign_detections(self, data):
        """
        Assign Detections on a per-frame level

        :param data: The Data (two dataframes)
        :return:
        """
        det_, obj_ = data

        # Iterate over detections
        det_[self.__id_col] = None
        for fno, fdets in det_.groupby(level=0):
            if fno in obj_:  # Can only continue if we have objects visible
                # Extract Weights
                bbs, obj = fdets.droplevel(0)[self.__bb_col], obj_.loc[fno].astype(int)
                f_w = self.__dist_obj.weights(bbs.values, obj.values)
                # --- Now Branch on whether this is a Full-Fledged Model or not --- #
                if self.__dist_obj.models_hidden:
                    sI, sJ = len(bbs), len(obj)
                    mdl = mip.Model(name="Identifier", sense=mip.MAXIMIZE)
                    # Solve Model
                    a = [
                        [mdl.add_var(var_type=mip.BINARY) for _ in range(sJ + 1)]
                        for __ in range(sI + 1)
                    ]
                    for i in range(sI):
                        mdl += mip.xsum(a[i][j] for j in range(sJ + 1)) == 1
                    for j in range(sJ):
                        mdl += mip.xsum(a[i][j] for i in range(sI + 1)) == 1
                    mdl += a[-1][-1] == 0
                    mdl.objective = mip.maximize(
                        mip.xsum(f_w[i, j] * a[i][j] for i in range(sI + 1) for j in range(sJ + 1))
                    )
                    # Format Output
                    if mdl.optimize(2) in (
                        mip.OptimizationStatus.OPTIMAL,
                        mip.OptimizationStatus.FEASIBLE,
                    ):
                        for (i, j) in it.product(range(sI), range(sJ)):
                            if abs(a[i][j].x) > 0.5:
                                det_.loc[(fno, bbs.index[i]), self.__id_col] = obj.index[j]
                else:
                    for i, j in zip(*npext.hungarian(f_w, maximise=True)):
                        det_.loc[(fno, bbs.index[i]), self.__id_col] = obj.index[j]

        return det_[[self.__id_col]]


class OracleIdentifier:
    """
    This Class implements the Oracle-scheme for Gold-Standard Identification.

    The OracleIdentifier implements a scheme for obtaining 'Gold-Standard' Identification of
    detections given annotated bounding boxes at specific intervals, as described in MISC_056.

    @TODO Ensure that works correctly when I have one or both of Detections/Annotations empty
    """

    @staticmethod
    def remove_tunnel(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience Method to remove entries which contain Tunnel Annotations
        :param df: DataFrame
        :return: DataFrame
        """
        df = df.stack(0)
        df = df.loc[~df["GT.ID"].str.contains("T"), ["GT.BB", "GT.Detect", "GT.Type", "GT.ID"]]
        return df.unstack(-1).reorder_levels((1, 0), axis=1)

    def __init__(self, iou_min=0.75, iou_hard=0.5, bb_col="Det.BB", id_col="GS.ID"):
        """
        Initialises the Oracle.
        """
        self.__iou_min = iou_min
        self.__iou_hrd_min = iou_hard
        self.__bb_col = bb_col
        self.__id_col = id_col

    def identify(self, dets_df, annot_df):
        """
        Identifies the Tracklets

        The method assigns identities to Detections. This is based on IoU between corresponding
        BBoxes. For further details of assignment, see MISC_056 (pg2)

        :param dets_df: DataFrame containing BBoxes from different detections. The index
            indicates the Frame, while the columns are two-level: Level 0 is the Detection ID,
            while L1 must have at least one column named bb_col.
        :param annot_df: DataFrame of annotations, containing BBoxes from different identities
            (along the columns) with time (frames) along the rows. The index will be used as the
            prototype for the results. The columns should be triply-indexed, with level 0 indicating
            the identity, level 1 classed as Single, Huddle or Tentative(.1/.2), and level 2
            listing the BBox ('GT.BB'), and whether it is Hard to detect ('GT.Detect').
        :return: Identified Detections. Each Detection is assigned a set of identities, indicated by
            the string within the id_col. The Output index follows the format of dets_df,
            albeit with the different detections as a secondary index level.
        """
        # ==== Join ==== #
        data = pd.concat([annot_df, dets_df], axis=1, keys=["A", "D"], join="inner")

        # ==== Assign on a per-Frame Basis ==== #
        idDf = []
        for fIdx, fData in data.iterrows():
            # Extract Data
            ann = fData["A"].unstack(-1).dropna(how="all")
            det = fData["D"].unstack(-1).dropna(how="all")
            # Prepare List of Annotations
            a_box, a_idx, a_iou = [], [], []
            for i, a in ann.iterrows():
                if a["GT.Type"] == VIAParser.TYPE_HUDDLE:
                    for _ in a["GT.ID"]:
                        a_box.append(a["GT.BB"])
                        a_idx.append(i)
                        a_iou.append(self.__iou_hrd_min)
                else:
                    a_box.append(a["GT.BB"])
                    a_idx.append(i)
                    a_iou.append(self.__iou_min if a["GT.Detect"] else self.__iou_hrd_min)
            # Prepare list of Detections
            d_box, d_idx = det[self.__bb_col].values, det.index
            # Compute Distance (IoU)
            iou = cvext.pairwise_iou(d_box, a_box)
            iou[iou < np.array(a_iou, ndmin=2)] = np.NINF  # Cutoff
            # Perform Optimisation
            assigned = npext.hungarian(iou, maximise=True, row_labels=d_idx, col_labels=a_idx)
            # Update Detection List
            det[self.__id_col] = ""
            for d, a in zip(*assigned):
                det.loc[d, self.__id_col] = ann.loc[a, "GT.ID"]
            idDf.append(det)

        # ==== Build DataFrame and return ==== #
        return pd.concat(idDf, keys=data.index).rename_axis(index={None: "Det.ID"})

