package com.fugui.carpal;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;

import ai.onnxruntime.OrtException;

public class MainActivity extends AppCompatActivity implements DetectionCallback {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION};

    private CameraController cameraController;
    private VehicleDetector vehicleDetector;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);

        try {
            // IMPORTANT: Replace "yolo_model.onnx" with the actual name of your model file in the assets folder.
            vehicleDetector = new VehicleDetector(this, "yolo11m.onnx");
        } catch (OrtException | IOException e) {
            Log.e(TAG, "Error initializing VehicleDetector", e);
            Toast.makeText(this, "Failed to load model.", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        PreviewView viewFinder = findViewById(R.id.viewFinder);
        cameraController = new CameraController(this, this, viewFinder, vehicleDetector, this);

        if (allPermissionsGranted()) {
            cameraController.startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                cameraController.startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraController != null) {
            cameraController.stopCamera();
        }
    }

    @Override
    public void onDetections(Bitmap imageWithDetections) {
        runOnUiThread(() -> imageView.setImageBitmap(imageWithDetections));
    }
}