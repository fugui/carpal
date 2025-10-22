package com.fugui.carpal;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.IOException;
import java.util.List;

import ai.onnxruntime.OrtException;

public class VehicleDetector {

    private static final String TAG = "VehicleDetector";
    private final Context context; // Store context for asset loading
    private final YoloModelDetector yoloDetector;

    public VehicleDetector(Context context, String modelPath) throws OrtException, IOException {
        this.context = context;
        this.yoloDetector = new YoloModelDetector( context.getAssets().open(modelPath) );
    }

    public List<DetectionResult> detect(Bitmap image) {
        List<DetectionResult> results = yoloDetector.detectFromBitmap(image);
        Log.i( TAG, "Detected " + results.size() + " Objects" );
        results.stream().forEach( o -> Log.i( TAG,  "Object " + o.getClassName() + " " + o.getConfidence() + " " + o.getBoundingBox().toString() ) );
        return results;
    }

}
