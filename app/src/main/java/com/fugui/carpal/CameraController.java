package com.fugui.carpal;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
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

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.InputStream;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

public class CameraController {

    private static final String TAG = "CameraController";
    private final Context context;
    private final LifecycleOwner lifecycleOwner;
    private final PreviewView previewView;
    private final VehicleDetector vehicleDetector;
    private final DetectionCallback detectionCallback;
    private ExecutorService cameraExecutor;

    private static Bitmap debugPicture;

    public CameraController(Context context, LifecycleOwner lifecycleOwner, PreviewView previewView, VehicleDetector vehicleDetector, DetectionCallback detectionCallback) {
        this.context = context;
        this.lifecycleOwner = lifecycleOwner;
        this.previewView = previewView;
        this.vehicleDetector = vehicleDetector;
        this.detectionCallback = detectionCallback;


        try (InputStream inputStream = context.getAssets().open("road1.jpeg")) {
            debugPicture = BitmapFactory.decodeStream(inputStream);
        } catch (Exception e) {
            debugPicture = null;
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

                imageAnalysis.setAnalyzer(cameraExecutor, new FrameAnalyzer(vehicleDetector, detectionCallback));

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

        public FrameAnalyzer(VehicleDetector vehicleDetector, DetectionCallback detectionCallback) {
            this.vehicleDetector = vehicleDetector;
            this.detectionCallback = detectionCallback;
        }

        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastAnalyzedTimestamp.get() >= ANALYSIS_INTERVAL_MS) {
                lastAnalyzedTimestamp.set(currentTime);

                Bitmap bitmap = imageProxy.toBitmap(); // debugImage
                if (bitmap != null) {

                    List<DetectionResult> detections = vehicleDetector.detect(bitmap);
                    textRecognizeDetections(bitmap, detections);

                }
            }
            imageProxy.close();
        }

        private void textRecognizeDetections(Bitmap bitmap, List<DetectionResult> detections) {

            TextRecognizer recognizer = TextRecognition.getClient(new ChineseTextRecognizerOptions.Builder().build());


            List<Task<Text>> listTasks = detections.stream().map(detection -> {
                        RectF rectF = detection.getBoundingBox();

                        InputImage image = InputImage.fromBitmap(
                                Bitmap.createBitmap(bitmap, (int) rectF.left, (int) rectF.top, (int) rectF.width(), (int) rectF.height()),
                                0);
                        return recognizer.process(image);
                    }
            ).collect(Collectors.toList());

            Tasks.whenAllComplete(listTasks)
                    .addOnCompleteListener(result -> {
                        if (!result.isSuccessful())
                            return;
                        List<Task<?>> completedTasks = result.getResult();

                        for (int i = 0; i < completedTasks.size(); i++) {
                            Text visionText = (Text) completedTasks.get(i).getResult();
                            String resultText = visionText.getText();
                            Log.i(TAG, "Recognized Text: " + resultText);
//                            for (Text.TextBlock block : visionText.getTextBlocks()) {
//                                String blockText = block.getText();
//                                Log.i(TAG, "TextBlock: " + blockText);
//                                for (Text.Line line : block.getLines()) {
//                                    String lineText = line.getText();
//                                    Log.i(TAG, "  Line: " + lineText);
//                                    for (Text.Element element : line.getElements()) {
//                                        String elementText = element.getText();
//                                        Log.i(TAG, "    Element: " + elementText);
//                                    }
//                                }
//                            }
                            detections.get(i).setText( String.join( " ", resultText.split("\n")));
                        }


                        detectionCallback.onDetections( drawDetections( bitmap, detections  ));
                    })
                    .addOnFailureListener(e -> Log.e(TAG, "Text recognition failed.", e));
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
