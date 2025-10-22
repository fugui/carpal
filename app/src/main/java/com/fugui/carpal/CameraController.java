package com.fugui.carpal;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.Image;
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
    private final Context context;
    private final LifecycleOwner lifecycleOwner;
    private final PreviewView previewView;
    private final VehicleDetector vehicleDetector;
    private ExecutorService cameraExecutor;

    public CameraController(Context context, LifecycleOwner lifecycleOwner, PreviewView previewView, VehicleDetector vehicleDetector) {
        this.context = context;
        this.lifecycleOwner = lifecycleOwner;
        this.previewView = previewView;
        this.vehicleDetector = vehicleDetector;
    }

    public void startCamera() {
        cameraExecutor = Executors.newSingleThreadExecutor();
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(context);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, new FrameAnalyzer(vehicleDetector));

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis);

            } catch (Exception e) {
                Log.e(TAG, "Use case binding failed", e);
            }
        }, ContextCompat.getMainExecutor(context));
    }

    private static class FrameAnalyzer implements ImageAnalysis.Analyzer {
        private final VehicleDetector vehicleDetector;
        private final AtomicLong lastAnalyzedTimestamp = new AtomicLong(0);
        private static final long ANALYSIS_INTERVAL_MS = 2000; // 2 seconds

        public FrameAnalyzer(VehicleDetector vehicleDetector) {
            this.vehicleDetector = vehicleDetector;
        }

        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastAnalyzedTimestamp.get() >= ANALYSIS_INTERVAL_MS) {
                lastAnalyzedTimestamp.set(currentTime);

                Bitmap bitmap = imageProxy.toBitmap();
                if (bitmap != null) {
                    List<DetectionResult> detections = vehicleDetector.detect(bitmap);
                    if (!detections.isEmpty()) {
                        Log.d(TAG, "Detected " + detections.size() + " vehicles.");
                        // We will process these detections in the next steps.
                    }
                }
            }
            imageProxy.close();
        }
    }

    public void stopCamera() {
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }
}