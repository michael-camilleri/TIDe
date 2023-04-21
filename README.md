# Tracking and Identification Dataset (TIDe)

This repository contains the Tracking and Identification Dataset (TIDe for short) for group-housed mice as described in our paper “Persistent Animal Identification Leveraging Non-Visual Markers” [1] and further elaborated on in Chapters 3/4 of my PhD Thesis “Automated Identification and Behaviour Classification for Modelling Social Dynamics in Group-Housed Mice” [2].

In addition, we also make available code and video clips supporting the paper [1]. 

The Data is released under the CC-BY 4.0 License: Code is released under the GNU GPL v3.0 License. We provide copies of both license definitions as DATA.LICENSE and CODE.LICENSE respectively.


## 1. Repository Structure

***Note***: Due to size limitations, the [Detection **images** (todo)](todo) and [Identifier **Segments** (todo)](todo) information are stored as zip-files on our University of Edinburgh project pages respectively.

```
├── datasets
│    ├── Detections                    # Curated Detection Dataset
│    │    ├── Train.json
│    │    ├── Validate.json
│    │    ├── Test.json
│    │    └── images                   # Directory of JPEG frames (available separately through DataShare)
│    │         ├── 00000.jpg
│    │         :
│    │         └── 04585.jpg
│    │
│    └── Identifications               # Curated Identification Dataset
│         ├── Datasplits.df
│         ├── Annotations.df
│         └── Segments                 # Per-Segment Data (available separately through DataShare)
│              ├── A_17
│              │    ├── Positions.df
│              │    ├── Detections.df
│              │    └── Video.mp4
│              :
│              └── P_43
│
├── TIM
│    ├── code                          # Code implementations of TIM
│    │    ├── Detectors.py
│    │    ├── Trackers.py
│    │    ├── Identifiers.py
│    │    └── Evaluation.py
│    │
│    └── results                       # Video-Clips of the Identifier in action
│         ├── Clip_1.mp4
│         :
│         └── Clip_6.mp4
│
├── LICENSE
└── README.md
```

---------

## 2. Datasets
 * The TIDe dataset aims to support automatic tracking and identification of group-housed mice in the home-cage using non-visual (RFID) identity cues.
 * They key modalities are Video Frames and 2D coordinates picked up through an Antenna Grid: additionally, we have provided annotations of mouse bounding-boxes (BBoxes).
 * The data is organised into two subsets, one pertaining to anonymous mouse **Detections** (i.e. no absolute identities) and an **Identifications** subset which annotates a smaller subset of frames with mouse identity information.
 * For details of the data collection setup see [1, §5.1] (or [2, §3.1.2] for more detail).
 * The data is provided for reproducibility purposes and also to allow further research on the data: for this reason, it is provided in its complete form and with minimal processing.

### 2.1 Detections Data Subset
 * The ***Detections*** Subset contains annotated BBoxes of the visible mice in each frame, as well as a bounding box of the cardboard tunnel.
 * The data is stored in CoCo format [4] with some specific features (see below).
 * We provide 3247 from for the Training split, 570 for Validation and 769 for the Test split.
 * Further details about the data collection and annotation process appear in [2, §3.4.4]

#### 2.1.1 Storage Organisation
 * The dataset is stored partly within this repository (under `datasets/Detections`), with the images available through a dedicated [Detections Subset page on DataShare (Todo)](todo) as a zip-file: an empty *images* directory is provided to indicate where it should exist relative to the root.
 * The images from all three datasplits (Train/Validate/Test) are stored in the same directory and numbered sequentially from 0 to 4585 (name is with 5 leading zeros).
 * The annotations are stored separately for each datasplit.

#### 2.1.2 Groundtruth JSON format
 * For each frame, we provide an axis-aligned BBox for each visible mouse as well as the tunnel.
 * The annotated BBoxes follow the CoCo format [4], with the additional information:
    * For *images*:
        * `hard`: bool — If true, image contains at least one annotation marked as `hard` (see below)
        * `cage`: str  — Alphabetical code for the cage from which the observation came (same as in [2]).
    * For *annotations*:
        * `hard`: bool — If true, indicates that the mouse/tunnel is difficult to make out. This is a suggestion that such data need not be relied upon too heavily.

### 2.2 Identifications Data Subset
 * The ***Identifications*** Subset contains side-view IR videos of the home-cage and RFID-based positions of each of the three mice.
 * Additionally, we provide pre-generated Detections of the mice in the videos: as such, the identification process can be run without the need of a separate detector (and without the need to process the videos).
 * The basic unit of processing is the video-frame at 25FPS: these are organised into 30-minute long segments. There are multiple segments per-cage.
 * Mice are labelled *R*ed, *G*reen or *B*lue: this is arbitrary but consistent within the cage.
 * Annotations were carried out using VIA [5] at a 4s rate for selected three-minute snippets, yielding 753 annotated frames within the Train split, 573 for Validation and 834 for the Test split.
 * Further details of the annotation and curation process appear in [1, §A] — for a more detailed description, refer to [2, §3.4.2─3.4.3].

#### 2.2.1 Storage Organisation
 * The dataset is stored partly within this repository (under `datasets/Identifications`), with the per-segment information available through a dedicated [Identifications Subset page on DataShare (Todo)](todo) as a zip-file: an empty *Segments* directory is provided to indicate where it should exist relative to the root.
 * The annotations are stored as one pandas dataframe, `Annotations.df`: the rest of the data is grouped with the segment from which it originates.
 * Information about the datasplit (Train/Validate/Test) is provided in the `Datasplits.df` dataframe.
 * Segments are named according to the cage and the segment number within that cage. Within each, there is:
    * `Positions.df`: Pandas dataframe of per-mouse RFID-based Positions
    * `Detections.df`: Pandas dataframe of (anonymous) mouse Detections
    * `Video.mp4`: IR video recording


#### 2.2.2 Format of individual Dataframes
 * All dataframes are stored in compressed (bz2) pickle format, and can be retrieved into python using: `pd.read_pickle('...', compression='bz2')`.

#### `Datasplits.df`
 * Indexed by *Cage-ID* (alphabetical) and *Segment-ID* (numerical)
 * We provide two columns:
    * `Datasplit`: str — The split to which the segment is assigned: *Train*, *Validate* or *Test*
    * `Evaluation`: bool — Whether the segment is used in the end-to-end evaluation (some segments have little usable data due to the mice continuously huddling).

#### `Annotations.df`
 * Indexed by *Cage-ID*, *Segment-ID*, *Frame* (since start of segment) and *Annotation* (arbitrary alphabetical enumeration).
 * The columns are two-level. For each annotation we provide:
    * `GT.BB`: The annotated axis-aligned bounding box. This is provided in CoCo format as sub-columns — top-left (x,y) and size (w,h).
    * `GT.Metadat`: Metadata about the BBox:
        * `Type`: Indicates if it is a *Single* object, *Huddle* of mice or a *Tentative* identity.
        * `ID`: A single identity (R/G/B or T for Tunnel), or multiple thereof when type is Tentative/Huddle.
        * `Detectable`: To what extent the mouse is easy to make out (mirrors `Hard` flag in ***Detections*** subset)
        * `Occluded`: None (missing) or type of occlusion, as a comma separated list of occluders. Distinguish between empty string (Clear) and None (missing).
        * `Truncated`: Level of truncation: *Clear*,*Truncated* or *Marginal* (marginal is also for ignored)
        * `Split`: Whether the body of the mouse/tunnel is split by an occluder (bool).
        * `Covering`: None (missing) or whether the object is covering another *Mouse* or *Tunnel* (comma separated list). Distinguish between empty string (Not covering anything) and None (missing).
        * `Viewpoint`: None (missing) or *Sideways [T]*, *Frontal [T]* (both for the Tunnel), *Facing [M]*, *Back [M]*, *Right [M]*, *Left [M]* or *Vertical [M]* (for Mice).

#### `Positions.df`
 * Indexed by *Frame* (since segment start)
 * For each frame, we provide the position of each mouse (*R*, *G* and *B*) as captured using the RFID baseplate from the Actual Analytics HCA Rig (see [3])
 * The position is encoded as the antenna number (1 through 18) in a 3×6 grid.

#### `Detections.df`
 * Indexed by *Frame* and *Detection*:
    * We provide up to 5 detections per frame for the mouse class.
 * The columns are two-level. For each detection, we provide:
    * `Det.BB`: The axis-aligned BBox as detected using the FCOS Detector [6]. This is represented in CoCo format as sub-columns --- top-left (x,y) and size (w,h).
    * `Det.Score`: The confidence score provided by the FCOS detector (0─1).

-----------

## 3. Implementations
 * We provide the [python code](https://github.com/michael-camilleri/TIDe/tree/main/TIM/results) for the Detector and Tracker wrappers, as well as the Identifier implementation and some Evaluation routines.
 * Note that the code is provided 'as is' for research purposes and scrutiny of the algorithm:
    * It does not include the accompanying setup/frameworks: these will be added at a later date.
    * It requires dependence on the external [FCOS](https://github.com/tianzhi0549/FCOS), [SORT](https://github.com/abewley/sort) and [SiamMask](https://github.com/foolwood/SiamMask) codebases.

-----------

## 4. Tracking and Identification Video Clips
We present sample illustrative videos showing the BBox tracking each mouse as given by our identification method, and comparisons with other techniques in [data/results](https://github.com/michael-camilleri/TIDe/tree/main/TIM/results).

### 4.1 DataSet Sample

#### [Clip 1](https://github.com/michael-camilleri/TIDe/blob/main/TIM/results/Clip_1.mp4)
This shows a sample video-clip of 30s from our data. The original raw video appears on the left: on the right, we show the same video after being processed with [CLAHE](https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/) to make it more visible to human viewers, together with the RFID-based position of each mouse overlayed as coloured dots. Note that our methods all operate on the raw video and not on the CLAHE-filtered ones.

### 4.2 Sample Identifications
We present some illustrative video examples showing the BBox tracking each mouse as given by our identification method, and comparisons with other techniques.
Apart from the first clip, the videos are played at 5 FPS for clarity, and illustrate the four methods we are comparing in a four-window arrangement with Ours on the top-left, the baseline on the top-right, the static assignment with the probabilistic model bottom-left and the SiamMask tracking bottom-right.
The frame number is displayed in the top-right corner of each sub-window.
The videos are best viewed by manually stepping through the frames.

#### [Clip 2](https://github.com/michael-camilleri/TIDe/blob/main/TIM/results/Clip_2.mp4)
We start by showing a minute-long clip of our identifier in action, played at the nominal framerate of 25 FPS.
This clip illustrates the difficulty of our setup, with the mice interacting with each other in very close proximity (e.g. Blue and Green in frames 42375 to 42570) and often partially/fully occluded by cage elements such as the hopper (e.g. Red in frames 42430 to 42500) or the tunnel (e.g. Blue in frames 42990 to 43020).

#### [Clip 3](https://github.com/michael-camilleri/TIDe/blob/main/TIM/results/Clip_3.mp4)
This clip shows the ability of our model to handle occlusion.
Note for example how in frames 40594 to 40605, the relative arrangement of the RFID tags confuses the baseline (top-right), but the (static) probabilistic weight model is sufficient to reason about the occlusion dynamics and the three mice.
Temporal continuity does however add an advantage, as in the subsequent frames, 40607 to 40630, even the static assignment (bottom-left) mis-identifies the mice, mostly due to the lag in the RFID signal.
The SiamMask (bottom-right) fails to consistently track any of the mice, mostly because of occlusion and passage through tunnel which happens later on in the video (not shown here), and shows the need for reasoning about occlusions.

#### [Clip 4](https://github.com/michael-camilleri/TIDe/blob/main/TIM/results/Clip_4.mp4)
This clip shows the weight model successfully filtering out spurious detections.
For clarity, we show only the static assignment (left) and the baseline (right).
Note how due to a change in the RFID antenna for Green, its BBox often gets confused for noisy detections below the hopper (see e.g. frames 40867 to 40875): however, the weight model, and especially the outlier distribution, is able to reject these and assign the correct BBox.

#### [Clip 5](https://github.com/michael-camilleri/TIDe/blob/main/TIM/results/Clip_5.mp4)
This shows another difficult case involving mice interacting and hiding each other.
The SiamMask is unable to keep track of any of the mice consistently.
While our method does occasionally lose the red mouse when it is severely occluded (e.g. frame 40990) the baseline gets it completely wrong (mis-identifying green for red in e.g. frames 40990 to 41008), mostly due to a lag in the RFID which also trips the static assignment with our weight model.

#### [Clip 6](https://github.com/michael-camilleri/TIDe/blob/main/TIM/results/Clip_6.mp4)
Finally, this shows an interesting scenario comparing our method (left) to the SiamMask tracker (right).
The latter is certainly smoother in terms of the flow of BBox, with the tracking-by-detection approach understandably being more jerky.
However, SiamMask's inability to handle occlusion comes to the fore towards the end of the clip (frames 40693 to end), where the green track latches to the face of the blue mouse.

----------------

## References
 If you make use of this data, please cite our work, as below:

 > [1] M. P. J. Camilleri, L. Zhang, R. S. Bains, A. Zisserman, and C. K. I. Williams, “Persistent Object Identification Leveraging Non-Visual Markers,” CoRR (arXiv), cs.CV (2112.06809), Dec. 2021. [Available on arXiv](https://arxiv.org/pdf/2112.06809.pdf)

 > [2] M. P. J. Camilleri, “Automated Identification and Behaviour Classification for Modelling Social Dynamics in Group-Housed Mice,” PhD Thesis, University of Edinburgh, 2023.

 The video recordings of the mice are courtesy of the Mary Lyon Centre at MRC Harwell, as described in:
 > [3] R. S. Bains, H. L. Cater, R. R. Sillito, A. Chartsias, D. Sneddon, D. Concas, P. Keskivali-Bond, T. C. Lukins, S. Wells, A. Acevedo Arozena, P. M. Nolan, and J. D. Armstrong. “Analysis of Individual Mouse Activity in Group Housed Animals of Different Inbred Strains using a Novel Automated Home Cage Analysis System”. In: Frontiers in Behavioral Neuroscience 10 (106) (June 2016). [Available online](https://core.ac.uk/reader/82834260)

 Other references:
 > [4] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common Objects in Context. In: Computer Vision – ECCV 2014, pp. 740–755. Springer.

 > [5] A. Dutta and A. Zisserman, “The VIA Annotation Software for Images, Audio and Video,” in Proceedings of the 27th ACM International Conference on Multimedia, 2019.

 > [6] Z. Tian, C. Shen, H. Chen, and T. He, “FCOS: Fully Convolutional One-Stage Object Detection,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9626–9635.
