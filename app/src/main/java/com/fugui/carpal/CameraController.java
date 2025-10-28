package com.fugui.carpal;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
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

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

import ai.onnxruntime.OrtException;

public class CameraController {

    private static final String TAG = "CameraController";
    private final Context context;
    private final LifecycleOwner lifecycleOwner;
    private final PreviewView previewView;
    private final VehicleDetector vehicleDetector;
    private final DetectionCallback detectionCallback;
    private ExecutorService cameraExecutor;

    private static Bitmap debugPicture = null;

    public CameraController(Context context, LifecycleOwner lifecycleOwner, PreviewView previewView, VehicleDetector vehicleDetector, DetectionCallback detectionCallback) {
        this.context = context;
        this.lifecycleOwner = lifecycleOwner;
        this.previewView = previewView;
        this.vehicleDetector = vehicleDetector;
        this.detectionCallback = detectionCallback;


        try (InputStream inputStream = context.getAssets().open("road1.jpeg")) {
            debugPicture = BitmapFactory.decodeStream(inputStream);
        } catch (Exception ignored) {
        }
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

                imageAnalysis.setAnalyzer(cameraExecutor, new FrameAnalyzer(context, vehicleDetector, detectionCallback));

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
        private final DetectionCallback detectionCallback;
        private final AtomicLong lastAnalyzedTimestamp = new AtomicLong(0);
        private static final long ANALYSIS_INTERVAL_MS = 2000; // 2 seconds
        private final PaddleOrtEngine paddleOrtEngine;

        public FrameAnalyzer(Context context, VehicleDetector vehicleDetector, DetectionCallback detectionCallback) {
            this.vehicleDetector = vehicleDetector;
            this.detectionCallback = detectionCallback;

            PaddleOrtEngine temp = null;
            try {
                temp = new PaddleOrtEngine(context, "det.onnx", "cls.onnx", "rec.onnx", "dict.txt");
            } catch (IOException | OrtException e) {
                Log.e(TAG, "Error initializing PaddleOrtEngine", e);
            }
            paddleOrtEngine = temp;
            Log.i(TAG, "Created PaddleOrtEngine: " + paddleOrtEngine);

        }

        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastAnalyzedTimestamp.get() >= ANALYSIS_INTERVAL_MS) {
                lastAnalyzedTimestamp.set(currentTime);

                Bitmap bitmap = debugPicture; // imageProxy.toBitmap(); //
                if (bitmap != null) {

                    List<DetectionResult> detections = vehicleDetector.detect(bitmap);
                    textRecognizeDetections(bitmap, detections);

                }
                Log.i(TAG, "Totle analyze time: " + (System.currentTimeMillis() - currentTime));
            }
            imageProxy.close();
        }

        private void textRecognizeDetections(Bitmap bitmap, List<DetectionResult> detections) {

            detections.forEach(detection -> {
                Log.i(TAG, "OCR Start for : " + detection.getClassName());

                RectF rectF = detection.getBoundingBox();
                Bitmap oneCar = Bitmap.createBitmap(bitmap, (int) rectF.left, (int) rectF.top, (int) rectF.width(), (int) rectF.height());
                try {
                    PaddleOrtEngine.OcrResult ocrResult = paddleOrtEngine.runOcr(oneCar);

                    String text = ocrResult.texts.stream().collect(Collectors.joining(","));
                    Log.i(TAG, "OCR Result: " + text);

                    detection.setText(text);
                } catch (OrtException e) {
                    Log.e(TAG, "OCR Text recognize failed: ", e);
                }
            });

            detectionCallback.onDetections(drawDetections(bitmap, detections));
        }

        private Bitmap drawDetections(Bitmap bitmap, List<DetectionResult> detections) {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2.0f);
            paint.setTextSize(30.0f);

            for (DetectionResult detection : detections) {
                canvas.drawRect(detection.getBoundingBox(), paint);
                String label = detection.getClassName() + ": " + String.format("%.2f", detection.getConfidence());
                canvas.drawText(label, detection.getBoundingBox().left, detection.getBoundingBox().top - 10, paint);
                if (detection.getText() != null && !detection.getText().isEmpty())
                    canvas.drawText(detection.getText(), detection.getBoundingBox().left, detection.getBoundingBox().top + 25, paint);
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
