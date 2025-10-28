package com.fugui.carpal;

import ai.onnxruntime.*;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.*;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import java.io.*;
import java.nio.*;
import java.util.*;

public class PaddleOrtEngine implements Closeable {

    /* ========== 静态配置 ========== */
    private static final int[] DET_SHAPE = {1, 3, 736, 1280};
    private static final int[] CLS_SHAPE = {1, 3, 48, 192};
    private static final int[] REC_SHAPE = {1, 3, 48, 320};

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    private static final String TAG = "ocr";

    /* ========== 成员 ========== */
    private final OrtEnvironment env;
    private final OrtSession detSession, clsSession, recSession;
    private final List<String> labelList;
    private final Context context;

    /* ========== 构造 ========== */
    public PaddleOrtEngine(Context context,
                           String detPath, String clsPath,
                           String recPath, String dictPath)
            throws IOException, OrtException {
        this.context = context;
        AssetManager am = context.getAssets();
        env = OrtEnvironment.getEnvironment();
        detSession = createSession(am, detPath);
        clsSession = createSession(am, clsPath);
        recSession = createSession(am, recPath);
        labelList = loadDict(am, dictPath);
    }

    private OrtSession createSession(AssetManager am, String path)
            throws IOException, OrtException {
        byte[] raw = readAsset(am, path);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.addCPU(true);
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        return env.createSession(raw, opts);
    }

    /* ========== 1. 检测 ========== */
    public DetectResult detect(Bitmap src) throws OrtException {
        ResizeResult resizeResult = resizeKeepAspect(src, DET_SHAPE[2], DET_SHAPE[3]);
        Bitmap bmp = resizeResult.bitmap;

        try (OnnxTensor tensor = bitmapToTensor(bmp, DET_SHAPE)) {
            OrtSession.Result res = detSession.run(Map.of("x", tensor));
            float[][] probMap = ((float[][][][]) res.get(0).getValue())[0][0];
            List<RotatedBox> boxes = postDb(probMap, 0.3f, 0.5f);
            return new DetectResult(boxes, resizeResult.scale, resizeResult.padW, resizeResult.padH);
        } finally {
            if (bmp != null && !bmp.isRecycled()) {
                bmp.recycle();
            }
        }
    }

    private List<RotatedBox> postDb(float[][] prob, float thresh, float boxThresh) {
        int H = prob.length;
        int W = prob[0].length;

        boolean[][] bitmap = new boolean[H][W];
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                bitmap[i][j] = prob[i][j] > thresh;

        List<List<int[]>> contours = new ArrayList<>();
        boolean[][] vis = new boolean[H][W];
        int[] dx = {1, -1, 0, 0}, dy = {0, 0, 1, -1};
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                if (!bitmap[i][j] || vis[i][j]) continue;
                List<int[]> queue = new LinkedList<>();
                List<int[]> component = new ArrayList<>();
                queue.add(new int[]{i, j});
                vis[i][j] = true;
                while (!queue.isEmpty()) {
                    int[] p = queue.remove(0);
                    component.add(p);
                    for (int d = 0; d < 4; d++) {
                        int ni = p[0] + dx[d], nj = p[1] + dy[d];
                        if (ni < 0 || ni >= H || nj < 0 || nj >= W) continue;
                        if (!bitmap[ni][nj] || vis[ni][nj]) continue;
                        vis[ni][nj] = true;
                        queue.add(new int[]{ni, nj});
                    }
                }
                if (component.size() > 10) contours.add(component);
            }
        }

        List<RotatedBox> result = new ArrayList<>();
        for (List<int[]> pts : contours) {
            float score = 0;
            for (int[] p : pts) score += prob[p[0]][p[1]];
            score /= pts.size();
            if (score < boxThresh) continue;

            int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE;
            int maxX = Integer.MIN_VALUE, maxY = Integer.MIN_VALUE;
            for (int[] p : pts) {
                minX = Math.min(minX, p[1]);
                maxX = Math.max(maxX, p[1]);
                minY = Math.min(minY, p[0]);
                maxY = Math.max(maxY, p[0]);
            }

            int boxHeight = maxY - minY;
            // Reduced the padding to 50% of the height plus a small constant.
            int padding = (int) (boxHeight * 0.5) + 3;

            minX = Math.max(0, minX - padding);
            maxX = Math.min(W, maxX + padding);
            minY = Math.max(0, minY - padding);
            maxY = Math.min(H, maxY + padding);

            PointF[] pf = new PointF[4];
            pf[0] = new PointF(minX, minY);
            pf[1] = new PointF(maxX, minY);
            pf[2] = new PointF(maxX, maxY);
            pf[3] = new PointF(minX, maxY);
            result.add(new RotatedBox(pf, score));
        }
        return result;
    }

    /* ========== 2. 方向分类 & 3. 识别 ========== */
    public boolean isRotated180(Bitmap crop) throws OrtException {
        ResizeResult resizeResult = resizeKeepAspect(crop, CLS_SHAPE[2], CLS_SHAPE[3]);
        Bitmap bmp = resizeResult.bitmap;
        try (OnnxTensor tensor = bitmapToTensor(bmp, CLS_SHAPE)) {
            OrtSession.Result res = clsSession.run(Map.of("x", tensor));
            float[][] prob = (float[][]) res.get(0).getValue();
            return prob[0][1] > 0.5f;
        } finally {
            if (bmp != null && !bmp.isRecycled()) {
                bmp.recycle();
            }
        }
    }

    public String recognize(Bitmap crop) throws OrtException {
        ResizeResult resizeResult = resizeKeepAspect(crop, REC_SHAPE[2], REC_SHAPE[3]);
        Bitmap bmp = resizeResult.bitmap;
        try (OnnxTensor tensor = bitmapToTensor(bmp, REC_SHAPE)) {
            OrtSession.Result res = recSession.run(Map.of("x", tensor));
            float[][][] logits = (float[][][]) res.get(0).getValue();
            int[] pred = ctcDecode(logits[0]);
            return idxToStr(pred);
        } finally {
            if (bmp != null && !bmp.isRecycled()) {
                bmp.recycle();
            }
        }
    }

    /* ========== 完整端到端 ========== */
    public OcrResult runOcr(Bitmap src) throws OrtException {
        DetectResult detectResult = detect(src);
        List<RotatedBox> boxes = detectResult.boxes;
        List<String> texts = new ArrayList<>();

        int i = 0;
        for (RotatedBox b : boxes) {
            Bitmap crop = null;
            Bitmap rotatedCrop = null;
            try {
                crop = cropBox(src, b, detectResult.scale, detectResult.padW, detectResult.padH);

                Bitmap toRecognize = crop;
                if (isRotated180(crop)) {
                    rotatedCrop = rotate180(crop);
                    toRecognize = rotatedCrop;
                }

                texts.add(recognize(toRecognize));
            } finally {
                // Clean up all created bitmaps
                if (crop != null && !crop.isRecycled()) {
                    crop.recycle();
                }
                if (rotatedCrop != null && !rotatedCrop.isRecycled()) {
                    rotatedCrop.recycle();
                }
            }
        }
        return new OcrResult(boxes, texts);
    }

    /* ========== 工具 ========== */
    private OnnxTensor bitmapToTensor(Bitmap bmp, int[] shape) throws OrtException {
        int H = bmp.getHeight();
        int W = bmp.getWidth();
        float[] buf = new float[shape[1] * H * W];
        int[] pixels = new int[W * H];
        bmp.getPixels(pixels, 0, W, 0, 0, W, H);

        for (int j = 0; j < pixels.length; j++) {
            int p = pixels[j];
            float r = ((p >> 16) & 0xff) / 255.0f;
            float g = ((p >> 8) & 0xff) / 255.0f;
            float b = (p & 0xff) / 255.0f;
            buf[j] = (r - MEAN[0]) / STD[0];
            buf[H * W + j] = (g - MEAN[1]) / STD[1];
            buf[H * W * 2 + j] = (b - MEAN[2]) / STD[2];
        }

        FloatBuffer floatBuffer = FloatBuffer.wrap(buf);
        long[] longShape = Arrays.stream(shape).asLongStream().toArray();
        return OnnxTensor.createTensor(env, floatBuffer, longShape);
    }

    private ResizeResult resizeKeepAspect(Bitmap src, int tarH, int tarW) {
        float scale = Math.min((float) tarW / src.getWidth(), (float) tarH / src.getHeight());
        int scaledW = (int) (src.getWidth() * scale);
        int scaledH = (int) (src.getHeight() * scale);

        Bitmap scaledBmp = Bitmap.createScaledBitmap(src, scaledW, scaledH, true);
        Bitmap letterboxedBmp = Bitmap.createBitmap(tarW, tarH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(letterboxedBmp);

        int padW = (tarW - scaledW) / 2;
        int padH = (tarH - scaledH) / 2;

        canvas.drawBitmap(scaledBmp, padW, padH, null);
        scaledBmp.recycle();
        return new ResizeResult(letterboxedBmp, scale, padW, padH);
    }

    private Bitmap cropBox(Bitmap src, RotatedBox box, float scale, int padW, int padH) {
        Rect r = box.bound();
        int left = (int) ((r.left - padW) / scale);
        int top = (int) ((r.top - padH) / scale);
        int right = (int) ((r.right - padW) / scale);
        int bottom = (int) ((r.bottom - padH) / scale);

        left = Math.max(0, left);
        top = Math.max(0, top);
        right = Math.min(src.getWidth(), right);
        bottom = Math.min(src.getHeight(), bottom);

        int width = right - left;
        int height = bottom - top;

        if (width <= 0 || height <= 0) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888);
        }
        return Bitmap.createBitmap(src, left, top, width, height);
    }

    private Bitmap rotate180(Bitmap bmp) {
        Matrix m = new Matrix();
        m.postRotate(180);
        return Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
    }

    private int[] ctcDecode(float[][] prob) {
        List<Integer> idx = new ArrayList<>();
        int last = -1;
        for (float[] step : prob) {
            int maxIdx = 0;
            for (int i = 1; i < step.length; i++)
                if (step[i] > step[maxIdx]) maxIdx = i;
            if (maxIdx != 0 && maxIdx != last) idx.add(maxIdx);
            last = maxIdx;
        }
        return idx.stream().mapToInt(i -> i).toArray();
    }

    private String idxToStr(int[] idx) {
        StringBuilder sb = new StringBuilder();
        for (int i : idx) if (i < labelList.size()) sb.append(labelList.get(i));
        return sb.toString();
    }

    private List<String> loadDict(AssetManager am, String path) throws IOException {
        List<String> list = new ArrayList<>();
        list.add("blank");
        try (BufferedReader br = new BufferedReader(new InputStreamReader(am.open(path)))) {
            String line;
            while ((line = br.readLine()) != null) list.add(line.trim());
        }
        return list;
    }

    private byte[] readAsset(AssetManager am, String path) throws IOException {
        try (InputStream is = am.open(path)) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = is.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            return buffer.toByteArray();
        }
    }

    @Override
    public void close() throws IOException {
        try {
            detSession.close();
            clsSession.close();
            recSession.close();
            env.close();
        } catch (OrtException e) {
            throw new IOException(e);
        }
    }

    /* ========== 简单数据结构 ========== */
    private static class ResizeResult {
        final Bitmap bitmap;
        final float scale;
        final int padW, padH;

        public ResizeResult(Bitmap bitmap, float scale, int padW, int padH) {
            this.bitmap = bitmap;
            this.scale = scale;
            this.padW = padW;
            this.padH = padH;
        }
    }

    public static class DetectResult {
        public final List<RotatedBox> boxes;
        public final float scale;
        public final int padW, padH;

        public DetectResult(List<RotatedBox> boxes, float scale, int padW, int padH) {
            this.boxes = boxes;
            this.scale = scale;
            this.padW = padW;
            this.padH = padH;
        }
    }

    public static class RotatedBox {
        public final PointF[] pts;
        public final float score;

        public RotatedBox(PointF[] p, float s) {
            pts = p;
            score = s;
        }

        public Rect bound() {
            int left = (int) Math.min(Math.min(pts[0].x, pts[1].x), Math.min(pts[2].x, pts[3].x));
            int top = (int) Math.min(Math.min(pts[0].y, pts[1].y), Math.min(pts[2].y, pts[3].y));
            int right = (int) Math.max(Math.max(pts[0].x, pts[1].x), Math.max(pts[2].x, pts[3].x));
            int bot = (int) Math.max(Math.max(pts[0].y, pts[1].y), Math.max(pts[2].y, pts[3].y));
            return new Rect(left, top, right, bot);
        }
    }

    public static class OcrResult {
        public final List<RotatedBox> boxes;
        public final List<String> texts;

        public OcrResult(List<RotatedBox> b, List<String> t) {
            boxes = b;
            texts = t;
        }
    }
}
