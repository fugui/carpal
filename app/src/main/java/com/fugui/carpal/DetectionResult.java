package com.fugui.carpal;

import android.graphics.RectF;

public class DetectionResult {
    private final String className;
    private final float confidence;
    private final RectF boundingBox;
    private String text;

    public DetectionResult(String className, float confidence, RectF boundingBox) {
        this.className = className;
        this.confidence = confidence;
        this.boundingBox = boundingBox;
    }

    public String getClassName() {
        return className;
    }

    public float getConfidence() {
        return confidence;
    }

    public RectF getBoundingBox() {
        return boundingBox;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }
}