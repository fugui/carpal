package com.fugui.carpal;

import ai.onnxruntime.*;

import android.content.res.AssetManager;
import android.graphics.*;

import java.io.*;
import java.nio.*;
import java.util.*;

public class PaddleOrtEngine implements Closeable {

    /* ========== 静态配置 ========== */
    private static final int[] DET_SHAPE = {1, 3, 736, 1280};  // det 动态 shape 也可以
    private static final int[] CLS_SHAPE = {1, 3, 48, 192};
    private static final int[] REC_SHAPE = {1, 3, 48, 320};

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    /* ========== 成员 ========== */
    private final OrtEnvironment env;
    private final OrtSession detSession, clsSession, recSession;
    private final List<String> labelList;          // 字符表

    /* ========== 构造 ========== */
    public PaddleOrtEngine(AssetManager am,
                           String detPath, String clsPath,
                           String recPath, String dictPath)
            throws IOException, OrtException {
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
        opts.addCPU(true);                       //  void
        opts.setOptimizationLevel(
                OrtSession.SessionOptions.OptLevel.ALL_OPT); //  void
        return env.createSession(raw, opts);
    }

    /* ========== 1. 检测 ========== */
    /* ================= 1. 检测主函数 ================= */
    public List<RotatedBox> detect(Bitmap src) throws OrtException {
        // 1. 等比缩放至模型输入尺寸（736×1280）
        Bitmap bmp = resizeKeepAspect(src, DET_SHAPE[2], DET_SHAPE[3]);
        try (OnnxTensor tensor = bitmapToTensor(bmp, DET_SHAPE);   // shape [1,3,H,W]
             OrtSession.Result res = detSession.run(Map.of("x", tensor))) {

            // 2. 取概率图：DB 输出节点默认叫 "sigmoid_0.tmp_0"
            float[][] probMap = ((float[][][][]) res.get(0).getValue())[0][0]; // [H,W]

            // 3. 后处理 -> 多边形框
            return postDb(probMap, 0.3f, 0.5f);
        }
    }

    /* ================= 2. DB 后处理 ================= */
    private List<RotatedBox> postDb(float[][] prob,
                                    float thresh,
                                    float boxThresh) {
        int H = prob.length;
        int W = prob[0].length;

        /* 2.1 二值化 */
        boolean[][] bitmap = new boolean[H][W];
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                bitmap[i][j] = prob[i][j] > thresh;

        /* 2.2 连通域搜索（4-邻域） */
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
                if (component.size() > 50)   // 过滤极小噪声
                    contours.add(component);
            }
        }

        /* 2.3 最小面积矩形（简单版：直接求外接凸包 + 旋转卡壳可再升级） */
        List<RotatedBox> result = new ArrayList<>();
        for (List<int[]> pts : contours) {
            // 计算平均得分
            float score = 0;
            for (int[] p : pts) score += prob[p[0]][p[1]];
            score /= pts.size();
            if (score < boxThresh) continue;

            // 求外接矩形（这里用最简轴对齐矩形，生产环境可改用旋转矩形）
            int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE;
            int maxX = Integer.MIN_VALUE, maxY = Integer.MIN_VALUE;
            for (int[] p : pts) {
                minX = Math.min(minX, p[1]);
                maxX = Math.max(maxX, p[1]);
                minY = Math.min(minY, p[0]);
                maxY = Math.max(maxY, p[0]);
            }
            // 4 个点顺时针
            PointF[] pf = new PointF[4];
            pf[0] = new PointF(minX, minY);
            pf[1] = new PointF(maxX, minY);
            pf[2] = new PointF(maxX, maxY);
            pf[3] = new PointF(minX, maxY);
            result.add(new RotatedBox(pf, score));
        }
        return result;
    }

    /* ========== 2. 方向分类 ========== */
    public boolean isRotated180(Bitmap crop) throws OrtException {
        Bitmap bmp = Bitmap.createScaledBitmap(crop, CLS_SHAPE[3], CLS_SHAPE[2], true);
        try (OnnxTensor tensor = bitmapToTensor(bmp, CLS_SHAPE);   // [1,3,48,192]
             OrtSession.Result res = clsSession.run(Map.of("x", tensor))) {

            // 输出节点 softmax_0.tmp_0 形状 (1,2)
            float[][] prob = (float[][]) res.get(0).getValue();   // 只有 1 行
            return prob[0][1] > 0.5f;   // label=1 表示 180°
        }
    }

    /* ========== 3. 识别 ========== */
    public String recognize(Bitmap crop) throws OrtException {
        Bitmap bmp = resizeKeepAspect(crop, REC_SHAPE[2], REC_SHAPE[3]);
        try (OnnxTensor tensor = bitmapToTensor(bmp, REC_SHAPE);   // [1,3,48,320]
             OrtSession.Result res = recSession.run(Map.of("x", tensor))) {

            // 输出节点 save_infer_model/scale_0.tmp_0 形状 (1,L,6625)
            float[][][] logits = (float[][][]) res.get(0).getValue(); // [1][L][C]
            int[] pred = ctcDecode(logits[0]);   // 只要第 0 张图
            return idxToStr(pred);
        }
    }

    /* ========== 完整端到端 ========== */
    public OcrResult runOcr(Bitmap src) throws OrtException {
        List<RotatedBox> boxes = detect(src);
        List<String> texts = new ArrayList<>();
        for (RotatedBox b : boxes) {
            Bitmap crop = cropBox(src, b);
            if (isRotated180(crop)) {
                crop = rotate180(crop);
            }
            texts.add(recognize(crop));
        }
        return new OcrResult(boxes, texts);
    }

    /* ========== 工具 ========== */
    private OnnxTensor bitmapToTensor(Bitmap bmp, int[] shape) throws OrtException {
        int H = shape[2], W = shape[3];
        Bitmap rgb = Bitmap.createScaledBitmap(bmp, W, H, true);
        float[] buf = new float[shape[1] * H * W];
        int[] pixels = new int[W * H];
        rgb.getPixels(pixels, 0, W, 0, 0, W, H);

        // 归一化
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            float r = ((p >> 16) & 0xff) / 255.0f;
            float g = ((p >> 8) & 0xff) / 255.0f;
            float b = (p & 0xff) / 255.0f;
            buf[i] = (r - MEAN[0]) / STD[0];
            buf[H * W + i] = (g - MEAN[1]) / STD[1];
            buf[H * W * 2 + i] = (b - MEAN[2]) / STD[2];
        }

        long[] longShape = Arrays.stream(shape).asLongStream().toArray();

        // 1.23.1 可用：直接 ByteBuffer，无需 allocator 参数
        ByteBuffer bb = ByteBuffer.allocateDirect(buf.length * 4)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float f : buf) bb.putFloat(f);
        bb.flip();   // 复位 position

        return OnnxTensor.createTensor(env, bb, longShape, OnnxJavaType.FLOAT);
    }

    private List<String> loadDict(AssetManager am, String path) throws IOException {
        List<String> list = new ArrayList<>();
        list.add("blank");                 // CTC blank
        try (BufferedReader br = new BufferedReader(new InputStreamReader(am.open(path)))) {
            String line;
            while ((line = br.readLine()) != null) list.add(line.trim());
            return list;
        }
    }

    private byte[] readAsset(AssetManager am, String path) throws IOException {
        try (InputStream is = am.open(path)) {
            byte[] buf = new byte[is.available()];
            is.read(buf);
            return buf;
        }
    }

    private Bitmap resizeKeepAspect(Bitmap src, int tarH, int tarW) {
        float scale = Math.min(tarW / (float) src.getWidth(),
                tarH / (float) src.getHeight());
        int w = (int) (src.getWidth() * scale);
        int h = (int) (src.getHeight() * scale);
        return Bitmap.createScaledBitmap(src, w, h, true);
    }

    private Bitmap cropBox(Bitmap src, RotatedBox box) {
        // 简易版：先按水平外接矩形扣图，实际可写透视变换
        Rect r = box.bound();
        return Bitmap.createBitmap(src, r.left, r.top, r.width(), r.height());
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
    public static class RotatedBox {
        public final PointF[] pts;  // length=4，顺时针
        public final float score;

        public RotatedBox(PointF[] p, float s) {
            pts = p;
            score = s;
        }

        public Rect bound() {
            int left = (int) Math.min(Math.min(pts[0].x, pts[1].x),
                    Math.min(pts[2].x, pts[3].x));
            int top = (int) Math.min(Math.min(pts[0].y, pts[1].y),
                    Math.min(pts[2].y, pts[3].y));
            int right = (int) Math.max(Math.max(pts[0].x, pts[1].x),
                    Math.max(pts[2].x, pts[3].x));
            int bot = (int) Math.max(Math.max(pts[0].y, pts[1].y),
                    Math.max(pts[2].y, pts[3].y));
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