package com.fugui.carpal;

import android.graphics.RectF;

public class DetectionResult {
    private RectF boundingBox;
    private float confidence;
    private int classId;
    private String className;

    public DetectionResult(RectF boundingBox, float confidence, int classId, String className) {
        this.boundingBox = boundingBox;
        this.confidence = confidence;
        this.classId = classId;
        this.className = className;
    }

    // Getters
    public RectF getBoundingBox() { return boundingBox; }
    public float getConfidence() { return confidence; }
    public int getClassId() { return classId; }
    public String getClassName() { return className; }
}