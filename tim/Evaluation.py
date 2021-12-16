# Implements metrics for evaluating tracking
from mpctools.extensions import utils
import pandas as pd


class EvaluateFrmDetections:
    """
    Class for scoring models based on Detections (i.e. as per 3.1 in MISC_056)
    """

    @staticmethod
    def canonical_ids(ids_df, d_col="Det.ID", o_col="Obj.ID"):
        """
        Converts Identity Tracklets to Canonical form (including filling NA with empty string)

        :param ids_df: Identities DataFrame (typically indexed by Frame and Tracklets)
        :param d_col: Name of Column containing Detection ID
        :param o_col: Name of Column containing Assigned Object ID
        :return:
        """
        ids_df = ids_df.copy(deep=True)  # Make sure it is not modified
        ids_df[o_col] = ids_df[o_col].fillna("")
        return ids_df.reset_index(level=1, drop=True).set_index(d_col, append=True)

    def __init__(self, id_col="Obj.ID", gs_col="GS.ID"):
        """
        Initialises the Evaluator

        :param id_col:
        :param gs_col:
        :param dt_col:
        """
        self.__id_col = id_col
        self.__gs_col = gs_col
        self.__stats = []
        self.__scores = {
            'Accuracy': [],
            'Mis-ID': [],
            'FNR': [],
            'FPR': []
        }
        self.__method = {
            'Accuracy': self.__bb_accuracy,
            'Mis-ID': self.__bb_idswitch,
            'FNR': self.__bb_miss,
            'FPR': self.__bb_falseid,
        }
        self.__divisor = {
            'Accuracy': 'Samples',
            'Mis-ID': 'Detections',
            'FNR': 'Detections',
            'FPR': 'Background'
        }

    def score(self, ids, gs, name=None):
        """
        Score a single segment of Gold-Standard and Identities and store internally.

        :param ids: Dictionary of assigned Identities for each detection under each named model.
        Each dataframe is doubly indexed with Frame/Detection and at least one column named id_col.
        :param gs: Gold-Standard identities for each detection. Doubly indexed with
        Frame/Detection and at least one column named gs_col.
        :return: self, for chaining
        """
        # Setup Name
        name = utils.default(name, len(self.__stats))

        # Stats
        self.__stats.append(pd.Series(
            {
                'Samples': len(gs),
                'Detections': (gs[self.__gs_col].str.len() > 0).sum(),
                'Background': (gs[self.__gs_col].str.len() == 0).sum()
            },
            name=name
        ))

        scores = {s: {} for s in self.__scores.keys()}
        for mdl, mids in ids.items():
            _joined = gs.join(mids).fillna("")
            for s, mtd in self.__method.items():
                scores[s][mdl] = _joined.apply(mtd, axis=1).astype(float).sum()
        for s, score in scores.items():
            self.__scores[s].append(pd.Series(score, name=name))
        return self

    def summarise(self, metric=None, rate=True):
        """
        Summarise all scores over the total dataset.

        If metric is None, then this is a summary overall: otherwise, it gives detailed breakdown
        over a particular metric.

        :return: Pandas DataFrame containing Results
        """
        # Summarise Statistics
        stats = pd.concat(self.__stats, axis=1)
        stats['Total'] = stats.sum(axis=1)

        if metric is None:
            scores = []
            for name, score in self.__scores.items():
                score = pd.concat(score, axis=1).sum(axis=1)
                if rate:
                    score /= stats.loc[self.__divisor[name], 'Total']
                scores.append(score.rename(name))
            return pd.concat(scores, axis=1)
        elif metric.lower() == 'stats':
            return stats
        else:
            score = pd.concat(self.__scores[metric], axis=1)
            score['Total'] = score.sum(axis=1)
            if rate:
                score /= stats.loc[self.__divisor[metric]]
            return score

    def __bb_accuracy(self, det):
        """
        Computes the ID-Accuracy for a single detection at a single frame (Eq. 10)

        :param det: Detection Row, with two columns (string in each case, empty if not assigned).
        """
        return ((len(det[self.__id_col]) == 0) and (len(det[self.__gs_col]) == 0)) or (
            (len(det[self.__id_col]) != 0)
            and (len(det[self.__gs_col]) != 0)
            and (det[self.__id_col] in det[self.__gs_col])
        )

    def __bb_idswitch(self, det):
        """
        Returns True if Identity switch (i.e. ID != GS if both are not empty)
        """
        return (len(det[self.__id_col]) > 0) and (len(det[self.__gs_col]) > 0) and (det[self.__id_col] not in det[self.__gs_col])

    def __bb_miss(self, det):
        """
        Returns True if there is a miss (i.e. ID is empty when GS is not)
        """
        return (len(det[self.__gs_col]) > 0) and (len(det[self.__id_col]) == 0)

    def __bb_falseid(self, det):
        """
        Returns True if GS is empty but an ID is assigned
        """
        return (len(det[self.__id_col]) > 0) and (len(det[self.__gs_col]) == 0)


class EvaluateFrmImages:
    """
    This class implements the End-to-End Evaluation as described in the BMVC Paper.
    """

    @staticmethod
    def canonical_gts(gts_df):
        """
        """
        return (
            gts_df.droplevel(1, axis=1)
            .stack(0, dropna=False)[["GT.BB", "GT.Detect"]]
            .rename_axis(["Frame", "Object"])
        )

    @staticmethod
    def canonical_ids(ids_df, id_col="Obj.ID"):
        """
        Converts ID Dataframe to canonical form

        :param ids_df: Object DataFrame as returned by Identifier.
        :param id_col: Column containing ID
        :return: Modified DataFrame
        """
        return (
            ids_df.dropna(subset=[id_col])  # Remove Unassigned BBoxes
            .set_index(id_col, append=True)
            .reset_index(1, drop=True)
            .rename_axis(["Frame", "Object"])
        )

    def __init__(
        self, iou_min=0.5, iou_hard=0.3, ann_col="GT.BB", id_col="Trk.BB.Det", hrd_col="GT.Detect"
    ):
        """
        Initialises the Evaluator

        :param ann_col: Column name containing Annotated Ground-truth
        :param id_col: Column name containing the Identifier BB
        :param detect_col: Column name containing the Difficulty flag.
        """
        self.__iou = iou_min
        self.__iou_hrd = iou_hard
        self.__a = ann_col
        self.__o = id_col
        self.__d = hrd_col
        self.__stats = []
        self.__scores = {
            'Accuracy': [],
            'IoU': [],
            'FNR': [],
            'FPR': []
        }
        self.__method = {
            'Accuracy': self._img_accuracy,
            'IoU': self._img_iou,
            'FNR': self._img_miss,
            'FPR': self._img_falseid,
        }
        self.__divisor = {
            'Accuracy': 'Samples',
            'IoU': 'Visible',
            'FNR': 'Visible',
            'FPR': 'Hidden'
        }

    def score(self, ids, gts, name=None):
        """
        Score a single segment of Ground-truths and Identities and store internally.

        :param ids: The Assigned BBox for each object. Doubly indexed with Frame/Object and at
            least one column named id_col.
        :param gts: Ground-truth BBox for each object. Doubly indexed with Frame/Object and at
            least two columns named ann_col & hrd_col.
        :return: self, for chaining
        """
        # Setup Name
        name = utils.default(name, len(self.__stats))

        # Stats
        self.__stats.append(pd.Series(
            {
                'Samples': len(gts),
                'Visible': gts[self.__a].count(),
                'Hidden': gts[self.__a].isna().sum(),
            },
            name=name
        ))

        scores = {s: {} for s in self.__scores.keys()}
        for mdl, mids in ids.items():
            _joined = gts.join(mids)
            for s, mtd in self.__method.items():
                scores[s][mdl] = _joined.apply(mtd, axis=1).astype(float).sum()
        for s, score in scores.items():
            self.__scores[s].append(pd.Series(score, name=name))
        return self

    def summarise(self, metric=None, rate=True):
        """
        Summarise all scores over the total dataset.

        If metric is None, then this is a summary overall: otherwise, it gives detailed breakdown
        over a particular metric.

        :return: Pandas DataFrame containing Results
        """
        # Summarise Statistics
        stats = pd.concat(self.__stats, axis=1)
        stats['Total'] = stats.sum(axis=1)

        if metric is None:
            scores = []
            for name, score in self.__scores.items():
                score = pd.concat(score, axis=1).sum(axis=1)
                if rate:
                    score /= stats.loc[self.__divisor[name], 'Total']
                scores.append(score.rename(name))
            return pd.concat(scores, axis=1)
        elif metric.lower() == 'stats':
            return stats
        else:
            score = pd.concat(self.__scores[metric], axis=1)
            score['Total'] = score.sum(axis=1)
            if rate:
                score /= stats.loc[self.__divisor[metric]]
            return score

    def _img_accuracy(self, obj):
        """
        Computes the Image-Accuracy
        :param obj:
        :return:
        """
        iou = self.__iou if obj[self.__d] else self.__iou_hrd
        return (pd.isna(obj[self.__a]) and pd.isna(obj[self.__o])) or (
            pd.notna(obj[self.__a])
            and pd.notna(obj[self.__o])
            and (obj[self.__a].iou(obj[self.__o]) >= iou)
        )

    def _img_iou(self, obj):
        """
        Computes the IoU between predicted and actual BBox
        """
        if pd.notna(obj[self.__a]):
            return obj[self.__a].iou(obj[self.__o]) if pd.notna(obj[self.__o]) else 0
        else:
            return 0

    def _img_miss(self, obj):
        """
        Returns True if IoU between actual and predicted is too low.
        """
        iou = self.__iou if obj[self.__d] else self.__iou_hrd
        return pd.notna(obj[self.__a]) and (
            pd.isna(obj[self.__o]) or (obj[self.__a].iou(obj[self.__o]) < iou)
        )

    def _img_falseid(self, obj):
        """
        Returns True if a mouse is hidden but a detection is assigned.
        """
        return pd.isna(obj[self.__a]) and pd.notna(obj[self.__o])


