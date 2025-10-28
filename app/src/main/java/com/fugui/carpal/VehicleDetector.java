package com.fugui.carpal;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import ai.onnxruntime.OrtException;

public class VehicleDetector {

    private static final String TAG = "VehicleDetector";
    private final YoloModelDetector yoloDetector;
    private final PaddleOrtEngine paddleEngine;

    public VehicleDetector(Context context, String yoloModelPath) throws OrtException, IOException {
        this.yoloDetector = new YoloModelDetector(context.getAssets().open(yoloModelPath));
        // Initialize PaddleEngine here, assuming model files are in assets
        this.paddleEngine = new PaddleOrtEngine(context, "det.onnx", "cls.onnx", "rec.onnx", "dict.txt");
    }

    public List<DetectionResult> detect(Bitmap image, boolean recognizeText) {
        // 1. Detect vehicles using YOLO
        List<DetectionResult> detections = yoloDetector.detectFromBitmap(image);
        Log.i(TAG, "Detected " + detections.size() + " potential vehicles.");

        if (recognizeText) {
            // 2. For each detected vehicle, run OCR to find text
            for (DetectionResult detection : detections) {
                Bitmap vehicleBitmap = null;
                try {
                    RectF box = detection.getBoundingBox();
                    // Ensure the crop area is valid
                    if (box.left < 0 || box.top < 0 || box.right > image.getWidth() || box.bottom > image.getHeight()) {
                        Log.w(TAG, "Skipping invalid bounding box for OCR: " + box);
                        continue;
                    }
                    vehicleBitmap = Bitmap.createBitmap(image, (int) box.left, (int) box.top, (int) box.width(), (int) box.height());
                    
                    PaddleOrtEngine.OcrResult ocrResult = paddleEngine.runOcr(vehicleBitmap);
                    String recognizedText = ocrResult.texts.stream().collect(Collectors.joining(", "));
                    
                    if (!recognizedText.isEmpty()) {
                        Log.i(TAG, "OCR Result for vehicle: " + recognizedText);
                        detection.setText(recognizedText);
                    }
                } catch (OrtException e) {
                    Log.e(TAG, "OCR failed for a vehicle.", e);
                } finally {
                    if (vehicleBitmap != null && !vehicleBitmap.isRecycled()) {
                        vehicleBitmap.recycle();
                    }
                }
            }
        }

        return detections;
    }
}