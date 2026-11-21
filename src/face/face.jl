# Face detection, tracking and skin-region masks (M1).
# TODO: face detector (OpenCV.jl YuNet, Haar cascade fallback; MediaPipe via PythonCall later)
# TODO: tracker between detections (detect every N frames)
# TODO: skin ROI masks: forehead + both cheeks
# TODO: roi_mean_rgb(frame, mask) -> (R, G, B) averaged over the region
