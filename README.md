# TIM (Tracking and long-term Identification using non-visual Markers)

This repository contains Python code and Supplementary material (including Video Clips) supporting our paper:
> Michael P.J. Camilleri, Li Zhang, Andrew Zisserman and Christopher K.I. Williams, "Tracking and Long-Term Identification Using Non-Visual Markers", arXiv preprint [cs.CV], 2112.06809

The paper is available [here](https://arxiv.org/pdf/2112.06809.pdf)

## Repository Structure
```
├── tim
    ├── Evaluation.py   # Implementation of Evaluation functions
    ├── Trackers.py     # Wrapper around SORT Tracker
    └── Identifiers.py  # Implementation of ILP Identifier
├── data
│   └── results         # Video-Clips of the Identifier in action
├── LICENSE
└── README.md
```

## Implementations
 * The code is provided as is for research purposes and scrutiny of the algorithm, but is not guaranteed to work without the accompanying setup/frameworks.
 * In particular, note dependency on external [SORT](https://github.com/abewley/sort) repository for tracking.

## Tracking and Identification Video Clips
We present sample illustrative videos showing the Bounding Box (BBox) tracking each mouse as given by our identification method, and comparisons with other techniques in [data/results](https://github.com/michael-camilleri/TIM/data/results).

### DataSet Sample

#### [Clip 1](https://github.com/michael-camilleri/TIM/data/results/Clip_1.mp4) 
This shows a sample video-clip of 30s from our data. The original raw video appears on the left: on the right, we show the same video after being processed with [CLAHE](https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/) to make it more visible to human viewers, together with the RFID-based position of each mouse overlayed as coloured dots. Note that our methods all operate on the raw video and not on the CLAHE-filtered ones.

### Sample Identifications
We present some illustrative video examples showing the BBox tracking each mouse as given by our identification method, and comparisons with other techniques.
Apart from the first clip, the videos are played at 5 FPS for clarity, and illustrate the four methods we are comparing in a four-window arrangement with Ours on the top-left, the baseline on the top-right, the static assignment with the probabilistic model bottom-left and the SiamMask tracking bottom-right.
The frame number is displayed in the top-right corner of each sub-window.
The videos are best viewed by manually stepping through the frames.

#### [Clip 2](https://github.com/michael-camilleri/TIM/data/results/Clip_2.mp4)
We start by showing a minute-long clip of our identifier in action, played at the nominal framerate of 25 FPS.
This clip illustrates the difficulty of our setup, with the mice interacting with each other in very close proximity (e.g. Blue and Green in frames 42375 to 42570) and often partially/fully occluded by cage elements such as the hopper (e.g. Red in frames 42430 to 42500) or the tunnel (e.g. Blue in frames 42990 to 43020).

#### [Clip 3](https://github.com/michael-camilleri/TIM/data/results/Clip_3.mp4)
This clip shows the ability of our model to handle occlusion.
Note for example how in frames 40594 to 40605, the relative arrangement of the RFID tags confuses the baseline (top-right), but the (static) probabilistic weight model is sufficient to reason about the occlusion dynamics and the three mice.
Temporal continuity does however add an advantage, as in the subsequent frames, 40607 to 40630, even the static assignment (bottom-left) mis-identifies the mice, mostly due to the lag in the RFID signal.
The SiamMask (bottom-right) fails to consistently track any of the mice, mostly because of occlusion and passage through tunnel which happens later on in the video (not shown here), and shows the need for reasoning about occlusions.

#### [Clip 4](https://github.com/michael-camilleri/TIM/data/results/Clip_4.mp4)
This clip shows the weight model successfully filtering out spurious detections.
For clarity, we show only the static assignment (left) and the baseline (right).
Note how due to a change in the RFID antenna for Green, its BBox often gets confused for noisy detections below the hopper (see e.g. frames 40867 to 40875): however, the weight model, and especially the outlier distribution, is able to reject these and assign the correct BBox.

#### [Clip 5](https://github.com/michael-camilleri/TIM/data/results/Clip_5.mp4)
This shows another difficult case involving mice interacting and hiding each other.
The SiamMask is unable to keep track of any of the mice consistently.
While our method does occasionally lose the red mouse when it is severely occluded (e.g. frame 40990) the baseline gets it completely wrong (mis-identifying green for red in e.g. frames 40990 to 41008), mostly due to a lag in the RFID which also trips the static assignment with our weight model.

#### [Clip 6](https://github.com/michael-camilleri/TIM/data/results/Clip_6.mp4)
Finally, this shows an interesting scenario comparing our method (left) to the SiamMask tracker (right).
The latter is certainly smoother in terms of the flow of BBox, with the tracking-by-detection approach understandably being more jerky.
However, SiamMask's inability to handle occlusion comes to the fore towards the end of the clip (frames 40693 tp end), where the green track latches to the face of the blue mouse.

