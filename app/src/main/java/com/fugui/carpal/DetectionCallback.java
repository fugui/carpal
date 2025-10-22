package com.fugui.carpal;

import android.graphics.Bitmap;

/**
 * A callback interface to receive the results of the detection process.
 */
public interface DetectionCallback {
    /**
     * Called when a frame has been processed and detections are available.
     * @param imageWithDetections The original bitmap with detection results drawn on it.
     */
    void onDetections(Bitmap imageWithDetections);
}
