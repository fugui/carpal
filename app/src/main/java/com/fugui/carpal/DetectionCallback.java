package com.fugui.carpal;

import android.graphics.Bitmap;

import java.util.List;

/**
 * A callback interface to receive the results of the detection process.
 */
public interface DetectionCallback {
    /**
     * Called when a frame has been processed and detections are available.
     * @param imageWithDetections The original bitmap with detection results drawn on it.
     * @param detections A list of detection results with detailed information.
     */
    void onDetections(Bitmap imageWithDetections, List<DetectionResult> detections);
}
