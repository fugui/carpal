package com.fugui.carpal;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import ai.onnxruntime.OnnxModelMetadata;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class YoloModelDetector {
    private static final String TAG = "yolo";

    private final OrtSession ortSession;
    private final OrtEnvironment ortEnvironment;
    private final String[] labels;


    public YoloModelDetector(InputStream inputStream) throws IOException, OrtException {
        this.ortEnvironment = OrtEnvironment.getEnvironment();

        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];
        int len;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }
        byte[] modelBytes = byteBuffer.toByteArray();

        this.ortSession = ortEnvironment.createSession(modelBytes, new OrtSession.SessionOptions());

        // Fallback to hardcoded labels as the current ONNX runtime version might not support metadata reading.
        this.labels = extractLabelsFromMetadata();
        Log.i(TAG, "Loaded label: " + String.join(", ", labels));
    }

    public String[] extractLabelsFromMetadata() throws OrtException {
        // 尝试从模型元数据中获取标签信息
        OnnxModelMetadata metadata = ortSession.getMetadata();
        List<String> labelList = new ArrayList<>();

        // 方法1: 尝试从标准的元数据字段中获取
        if (metadata != null) {
            for (Map.Entry<String, String> entry : metadata.getCustomMetadata().entrySet()) {
                String key = entry.getKey().toLowerCase();
                String value = entry.getValue();

                if (key.contains("names") || key.contains("labels") || key.contains("classes")) {
                    // 尝试解析标签字符串
                    parseLabelsFromString(labelList, value);
                    break;
                }
            }
        }

        // 方法2: 如果元数据中没有找到，尝试从输出节点信息中获取
//        if (labels.isEmpty()) {
//            OrtSession.OutputMetadata outputMetadata = ortSession.getOutputInfo().values().iterator().next().getInfo();
//            // 这里可以根据输出节点的信息推断类别数量
//        }

        // 方法3: 如果仍然没有找到，使用默认的 COCO 标签作为后备
        if (labelList.isEmpty()) {
            labelList = getDefaultCOCOLabels();
        }

        return labelList.toArray(new String[0]);
    }

    private void parseLabelsFromString(List<String> labelList, String labelsString) {
        try {
            // 尝试不同的分隔符解析标签
            String[] parsedLabels;
            if (labelsString.contains(",")) {
                parsedLabels = labelsString.split(",");
            } else if (labelsString.contains(";")) {
                parsedLabels = labelsString.split(";");
            } else if (labelsString.contains("\n")) {
                parsedLabels = labelsString.split("\n");
            } else {
                // 尝试解析 JSON 格式
                if (labelsString.trim().startsWith("{")) {
                    parseLabelsFromJson(labelList, labelsString);
                    return;
                } else {
                    parsedLabels = new String[]{labelsString};
                }
            }

            // 清理标签
            for (String label : parsedLabels) {
                String cleanedLabel = label.trim()
                        .replace("\"", "")
                        .replace("'", "")
                        .replace("{", "")
                        .replace("}", "")
                        .replace("[", "")
                        .replace("]", "");
                if (!cleanedLabel.isEmpty()) {
                    labelList.add(cleanedLabel);
                }
            }
        } catch (Exception e) {
            labelList = getDefaultCOCOLabels();
        }
    }

    private void parseLabelsFromJson(List<String> labelList, String jsonString) {
        try {
            // 简单的 JSON 解析，处理类似 {"0": "person", "1": "bicycle"} 的格式
            String cleaned = jsonString.replace("{", "").replace("}", "").replace("\"", "").trim();
            String[] pairs = cleaned.split(",");

            // 创建临时列表并按索引排序
            Map<Integer, String> labelMap = new TreeMap<>();

            for (String pair : pairs) {
                String[] keyValue = pair.split(":");
                if (keyValue.length == 2) {
                    try {
                        int index = Integer.parseInt(keyValue[0].trim());
                        String label = keyValue[1].trim();
                        labelMap.put(index, label);
                    } catch (NumberFormatException e) {
                        // 忽略格式错误的条目
                    }
                }
            }

            labelList.addAll(labelMap.values());
        } catch (Exception e) {
            e.printStackTrace();
            labelList = getDefaultCOCOLabels();
        }
    }

    private List<String> getDefaultCOCOLabels() {
        return Arrays.asList(
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
        );
    }


    public List<DetectionResult> detectFromBitmap(Bitmap bitmap) {
        try {
            float[] inputTensor = preprocessImage(bitmap);
            return runInference(ortEnvironment, ortSession, labels, inputTensor, bitmap.getWidth(), bitmap.getHeight());
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    static private final int inputSize = 640;
    static private final float confidenceThreshold = 0.5f;
    static private final float nmsThreshold = 0.45f;

    private float[] preprocessImage(Bitmap bitmap) {
        // 调整图像大小
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);

        // 转换为浮点数组并归一化
        float[] input = new float[3 * inputSize * inputSize];
        int[] intValues = new int[inputSize * inputSize];

        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize);

        for (int i = 0; i < intValues.length; i++) {
            int pixel = intValues[i];
            // RGB 通道，归一化到 [0,1]
            input[i] = ((pixel >> 16 & 0xFF) / 255.0f);
            input[i + inputSize * inputSize] = ((pixel >> 8 & 0xFF) / 255.0f);
            input[i + 2 * inputSize * inputSize] = ((pixel & 0xFF) / 255.0f);
        }

        if (!resizedBitmap.equals(bitmap)) {
            resizedBitmap.recycle();
        }

        return input;
    }

    private List<DetectionResult> runInference(OrtEnvironment ortEnvironment, OrtSession ortSession,
                                               String[] labels,
                                               float[] input, int originalWidth, int originalHeight)
            throws Exception {
        String inputName = ortSession.getInputNames().iterator().next();
        long[] inputShape = {1, 3, inputSize, inputSize};

        FloatBuffer inputBuffer = FloatBuffer.wrap(input);
        OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, inputBuffer, inputShape);

        Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);

        try (OrtSession.Result output = ortSession.run(inputs)) {
            OnnxTensor outputTensor = (OnnxTensor) output.get(0);
            float[][][] outputData = (float[][][]) outputTensor.getValue();
            return postProcess(outputData, originalWidth, originalHeight, labels);
        } finally {
            inputTensor.close();
        }
    }

    private List<DetectionResult> postProcess(float[][][] outputs, int originalWidth, int originalHeight, String[] labels) {
        List<DetectionResult> results = new ArrayList<>();

        // YOLOv11 输出格式: [1, 84, 8400]
        float[][] output = outputs[0]; // 第一个批次

        for (int i = 0; i < output[0].length; i++) {
            // 获取所有检测框的置信度
            float maxScore = 0f;
            int classId = -1;

            // 从第5个元素开始是类别分数
            for (int j = 4; j < output.length; j++) {
                if (output[j][i] > maxScore) {
                    maxScore = output[j][i];
                    classId = j - 4;
                }
            }

            if (maxScore > confidenceThreshold && classId >= 0 && classId < labels.length) {
                // 解析边界框坐标 (cx, cy, w, h)
                float cx = output[0][i];
                float cy = output[1][i];
                float w = output[2][i];
                float h = output[3][i];

                // 转换为 (x1, y1, x2, y2) 格式
                float x1 = cx - w / 2;
                float y1 = cy - h / 2;
                float x2 = cx + w / 2;
                float y2 = cy + h / 2;

                // 缩放回原始图像尺寸
                float scaleX = originalWidth / (float) inputSize;
                float scaleY = originalHeight / (float) inputSize;

                RectF boundingBox = new RectF(
                        x1 * scaleX,
                        y1 * scaleY,
                        x2 * scaleX,
                        y2 * scaleY
                );

                String className = labels[classId];
                results.add(new DetectionResult(boundingBox, maxScore, classId, className));
            }
        }

        return nonMaxSuppression(results);
    }

    private List<DetectionResult> nonMaxSuppression(List<DetectionResult> detections) {
        // 按置信度排序
        detections.sort((d1, d2) -> Float.compare(d2.getConfidence(), d1.getConfidence()));
        List<DetectionResult> selected = new ArrayList<>();

        while (!detections.isEmpty()) {
            DetectionResult current = detections.get(0);
            selected.add(current);

            // 移除与当前检测框重叠度高的检测框
            List<DetectionResult> remaining = new ArrayList<>();
            for (int i = 1; i < detections.size(); i++) {
                DetectionResult detection = detections.get(i);
                if (iou(current.getBoundingBox(), detection.getBoundingBox()) < nmsThreshold) {
                    remaining.add(detection);
                }
            }
            detections = remaining;
        }

        return selected;
    }

    private float iou(RectF box1, RectF box2) {
        float intersectionLeft = Math.max(box1.left, box2.left);
        float intersectionTop = Math.max(box1.top, box2.top);
        float intersectionRight = Math.min(box1.right, box2.right);
        float intersectionBottom = Math.min(box1.bottom, box2.bottom);

        float intersectionArea = Math.max(0, intersectionRight - intersectionLeft) *
                Math.max(0, intersectionBottom - intersectionTop);

        float area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
        float area2 = (box2.right - box2.left) * (box2.bottom - box2.top);
        float unionArea = area1 + area2 - intersectionArea;

        return intersectionArea / unionArea;
    }

}
