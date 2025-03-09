# PupilDetect
Goal: Using live webcam footage, detect movement of the pupils

## So Far (2025-03-09)
- Using facial landmarks from live webcam footage, detect eye blinks
- Implement a facial landmark detector from **dlib** to identify regions of the eye (i.e. upper and lower eyelids)
- Calculate the Eye Aspect Ratio (EAR) to determine a baseline for whether the eyes are open or closed (uses the vertical distance btwn eyelids)

## Application
1) Abnormal pupil movement is a potential symptom of delirium
   Changes in pupil size (dilation); sluggish pupillary light reflex; rapid eye movement       (this involves the iris)
2) Gaze tracking: to observe the direction of a person's gaze
   Position of pupil with respect to an origin; angle of gaze
