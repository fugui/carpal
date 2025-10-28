package com.fugui.carpal;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

public class CameraController {

    private static final String TAG = "CameraController";
    private final LifecycleOwner lifecycleOwner;
    private final PreviewView previewView;
    private final VehicleDetector vehicleDetector;
    private final DetectionCallback detectionCallback;
    private ExecutorService cameraExecutor;

    public CameraController(Context context, LifecycleOwner lifecycleOwner, PreviewView previewView, VehicleDetector vehicleDetector, DetectionCallback detectionCallback) {
        this.lifecycleOwner = lifecycleOwner;
        this.previewView = previewView;
        this.vehicleDetector = vehicleDetector;
        this.detectionCallback = detectionCallback;
    }

    public void startCamera() {
        cameraExecutor = Executors.newSingleThreadExecutor();
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(previewView.getContext());

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, new FrameAnalyzer(vehicleDetector, detectionCallback));

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis);

            } catch (Exception e) {
                Log.e(TAG, "Use case binding failed", e);
            }
        }, ContextCompat.getMainExecutor(previewView.getContext()));
    }

    private static class FrameAnalyzer implements ImageAnalysis.Analyzer {
        private final VehicleDetector vehicleDetector;
        private final DetectionCallback detectionCallback;
        private final AtomicLong lastAnalyzedTimestamp = new AtomicLong(0);
        private static final long ANALYSIS_INTERVAL_MS = 2000; // 2 seconds

        public FrameAnalyzer(VehicleDetector vehicleDetector, DetectionCallback detectionCallback) {
            this.vehicleDetector = vehicleDetector;
            this.detectionCallback = detectionCallback;
        }

        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastAnalyzedTimestamp.get() < ANALYSIS_INTERVAL_MS) {
                imageProxy.close();
                return;
            }
            lastAnalyzedTimestamp.set(currentTime);

            Bitmap bitmap = imageProxy.toBitmap();
            if (bitmap != null) {
                Bitmap resultBitmap = null;
                List<DetectionResult> detections = null;
                try {
                    long startTime = System.currentTimeMillis();
                    detections = vehicleDetector.detect(bitmap, true); // Run with OCR
                    long endTime = System.currentTimeMillis();
                    Log.i(TAG, "vehicleDetector.detect duration: " + (endTime - startTime) + "ms");

                    resultBitmap = drawDetections(bitmap, detections);
                    detectionCallback.onDetections(resultBitmap, detections);
                } finally {
                    if (bitmap != null && !bitmap.isRecycled()) {
                        bitmap.recycle();
                    }
                }
            }
            imageProxy.close();
        }

        private Bitmap drawDetections(Bitmap originalBitmap, List<DetectionResult> detections) {
            Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(3.0f);
            paint.setTextSize(40.0f);
            paint.setTextAlign(Paint.Align.LEFT);

            Paint textBgPaint = new Paint();
            textBgPaint.setColor(Color.argb(150, 0, 0, 0)); // Semi-transparent black background
            textBgPaint.setStyle(Paint.Style.FILL);

            for (DetectionResult detection : detections) {
                canvas.drawRect(detection.getBoundingBox(), paint);

                String yoloLabel = detection.getClassName() + ": " + String.format("%.2f", detection.getConfidence());
                String ocrLabel = detection.getText();

                canvas.drawRect(detection.getBoundingBox().left, detection.getBoundingBox().top - 45,
                 detection.getBoundingBox().left + paint.measureText(yoloLabel), detection.getBoundingBox().top, textBgPaint);
                canvas.drawText(yoloLabel, detection.getBoundingBox().left, detection.getBoundingBox().top - 5, paint);

                if (ocrLabel != null && !ocrLabel.isEmpty()) {
                    canvas.drawRect(detection.getBoundingBox().left, detection.getBoundingBox().top,
                     detection.getBoundingBox().left + paint.measureText(ocrLabel), detection.getBoundingBox().top + 45, textBgPaint);
                    canvas.drawText(ocrLabel, detection.getBoundingBox().left, detection.getBoundingBox().top + 40, paint);
                }
            }
            return mutableBitmap;
        }
    }

    public void stopCamera() {
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }
}
