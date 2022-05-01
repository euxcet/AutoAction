package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
@RequiresApi(api = Build.VERSION_CODES.N)
public class EventAnalyzer {
    public static final String TAG = "EventAnalyzer";

    private Interpreter model;
    private Context context;
    private List<Map<String,Integer>> encodeMap;
    private float[] max;
    private float[] min;
    private long[] crtTimeStampS;
    private long[] crtEventTimeStampS;
    private float[][][] values;

    /** Load TF Lite model from asset file. */
//    public static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath)
//            throws IOException {
//        try (AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
//             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
//            FileChannel fileChannel = inputStream.getChannel();
//            long startOffset = fileDescriptor.getStartOffset();
//            long declaredLength = fileDescriptor.getDeclaredLength();
//            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
//        }
//    }

    public boolean loadMaxMin() {
        max = new float[29];
        min = new float[29];
        try {
            // InputStream inStream = context.getResources().openRawResource(R.raw.param_max);
            InputStream inStream = new FileInputStream(ContextActionContainer.getSavePath() + "param_max.json");
            BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
            String line;
            StringBuilder stringBuilder = new StringBuilder();
            while ((line = br.readLine()) != null) {
                stringBuilder.append(line);
            }
            br.close();
            inStream.close();
            JSONObject jsonObject = new JSONObject(stringBuilder.toString());
            JSONArray maxArray = jsonObject.getJSONArray("max");
            for (int i = 0; i < maxArray.length(); i++) {
                float fs = (float) maxArray.getDouble(i);
                max[i] = fs;
            }
            JSONArray minArray = jsonObject.getJSONArray("min");
            for (int i = 0; i < minArray.length(); i++) {
                float fs = (float) minArray.getDouble(i);
                min[i] = fs;
            }
            return true;
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return false;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public boolean loadEncoderMap() {
        encodeMap = new ArrayList<>();
        try {
            // InputStream inStream = context.getResources().openRawResource(R.raw.param_dicts);
            InputStream inStream = new FileInputStream(ContextActionContainer.getSavePath() + "param_dicts.json");
            BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
            String line;
            StringBuilder stringBuilder = new StringBuilder();
            while ((line = br.readLine()) != null) {
                stringBuilder.append(line);
            }
            br.close();
            inStream.close();
            JSONArray jsonArray = new JSONArray(stringBuilder.toString());
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject obj = jsonArray.getJSONObject(i);
                Map<String, Integer> map = new HashMap<>();
                Iterator<String> iterator = obj.keys();
                while (iterator.hasNext()) {
                    String key = (String) iterator.next();
                    Integer value = obj.getInt(key);
                    map.put(key, value);
                }
                encodeMap.add(map);
            }
            return true;
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return false;
    }

    private synchronized void loadModel() {
        try {
            Log.e(TAG,ContextActionContainer.getSavePath()+"ResultModel.tflite" );
//            ByteBuffer buffer = loadModelFile(this.context.getAssets(), ContextActionContainer.getSavePath()+"ResultModel.tflite");
            model = new Interpreter(new File(ContextActionContainer.getSavePath()+"ResultModel.tflite"));
            // model = new Interpreter(new File(BuildConfig.SAVE_PATH + "ResultModel.tflite"));
            Log.v(TAG, "TFLite model loaded.");

            boolean res = loadMaxMin();
            if (res) {
                Log.v(TAG, "MaxMin loaded.");
            }

            res = loadEncoderMap();
            if(res) {
                Log.v(TAG, "EncoderMap loaded.");
            }

        } catch (Exception ex) {
            Log.e(TAG, ex.getMessage());
        }
    }

    // 初始化使用 GPU 代理的解释器
//    GpuDelegate delegate = new GpuDelegate();
//    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
//    Interpreter interpreter;

    public void initialize(Context ctx) {
        context = ctx;
        model = null;
        loadModel();
        crtTimeStampS = new long[16];
        crtEventTimeStampS = new long[16];
        values = new float[1][16][29];
    }

    public float analyzerTest() {
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 16, 29}, DataType.FLOAT32);
        ByteBuffer buffer = ByteBuffer.allocate(4 * 16 * 29);

//        float[] floats =new float[]{0.9980469f,0.8181819f,0.5336189f,0f,0f,0f,1f,0.7692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533619f,0f,0f,0f,1f,0.7692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533619f,0f,0f,0f,1f,0.7692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533619f,0f,0f,0f,1f,0.61538464f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533619f,0f,0f,0f,1f,0.07692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.90909094f,0.53362024f,0.90000004f,0f,0f,0.7f,1f,0.07692308f,0.29955947f,0.4074074f,0f,0f,1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,1f,0.53362024f,0.90000004f,0f,0f,1f,1f,0.84102565f,0.57709247f,0.4074074f,0f,0f,1f,0f,0f,1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533652f,0f,0f,0f,1f,0.84615386f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533652f,0f,0f,0f,1f,0.7692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533652f,0f,0f,0f,1f,0.7692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.533652f,0f,0f,0f,1f,0.7692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,1f,0.5336913f,0.90000004f,0f,0f,1f,1f,0.84102565f,0.57709247f,0.4074074f,0f,0f,1f,0f,0f,1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.18181819f,0.5336925f,0.90000004f,0f,0f,1f,1f,0.11282051f,0.29955947f,0f,0f,0f,1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,1f,0.9980469f,0.8181819f,0.5337034f,0f,0f,0f,1f,0.84615386f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.8181819f,0.5337034f,0f,0f,0f,1f,0.07692308f,0f,0.29955947f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0.9980469f,0.90909094f,0.5337049f,0.90000004f,0f,0f,0.7f,1f,0.07692308f,0.29955947f,0.77160496f,0f,0f,1f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,0};
        float[][][] floats = new float[][][]{{
                {1.48508552e-04f, 9.09090936e-01f, 1.13346493e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 6.99999988e-01f, 1.00000000e+00f, 9.87755060e-01f, 2.57553965e-01f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.48508552e-04f, 9.09090936e-01f, 1.13276895e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.52044464e-04f, 9.09090936e-01f, 1.14216516e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.59116302e-04f, 9.09090936e-01f, 1.14738531e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.59116302e-04f, 9.09090936e-01f, 1.16339372e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.59116302e-04f, 9.09090936e-01f, 1.17000584e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.62652228e-04f, 9.09090936e-01f, 1.17104985e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 6.99999988e-01f, 1.00000000e+00f, 9.87755060e-01f, 2.57553965e-01f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.66188140e-04f, 9.09090936e-01f, 1.17522599e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.66188140e-04f, 9.09090936e-01f, 1.18392622e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.66188140e-04f, 9.09090936e-01f, 1.18879834e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.69724051e-04f, 9.09090936e-01f, 1.19541055e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.69724051e-04f, 9.09090936e-01f, 1.20202266e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.69724051e-04f, 9.09090936e-01f, 1.20793879e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 6.99999988e-01f, 1.00000000e+00f, 9.87755060e-01f, 2.57553965e-01f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.73259978e-04f, 9.09090936e-01f, 1.21281091e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.87403653e-04f, 9.09090936e-01f, 1.21803107e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f},
                {1.87403653e-04f, 9.09090936e-01f, 1.22151114e-02f, 5.88235319e-01f, 0.00000000e+00f, 0.00000000e+00f, 1.00000000e+00f, 1.00000000e+00f, 4.08163257e-02f, 2.57553965e-01f, 0.00000000e+00f, 4.65661287e-10f, 5.17401400e-10f, 1.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 4.71920706e-04f, 5.54077997e-05f, 9.25840213e-05f, 4.02301157e-05f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f}
        }};

//        for(float f:floats)
//        {
//            buffer.putFloat(f);
//        }
        inputFeature0.loadBuffer(buffer);
        // Runs model inference and gets result.
        TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 1}, DataType.FLOAT32);
        float[][] out = new float[][]{{-1}};
        ByteBuffer bufferout = ByteBuffer.allocate(4);
        outputFeature0.loadBuffer(bufferout);
        model.run(floats, out);
        return out[0][0];
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public float analyze(String eventString) {
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 16, 29}, DataType.FLOAT32);
        ByteBuffer buffer = ByteBuffer.allocate(4 * 16 * 29);

        long[] features = parseEventString(eventString);
        if (features == null)
            return 0;

        updateValues(features);

        float[][][] input = values.clone();
        inputFeature0.loadBuffer(buffer);
        TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1}, DataType.FLOAT32);
        float[][] out = new float[][]{{-1}};
        model.run(input, out);
        return out[0][0];
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public long[] parseEventString(String str) {
        int index = 0;
        long[] res = new long[29];
        try {
            str = str.replace("[ ClassName", ";ClassName");
            str = str.replace(" ]", "");
            str = str.replace("&apos;", "&apos");
            str = str.replace("&quot;", "&quot");
            String strings[] = str.split(";");

            if (strings.length == 31) {
                for (int i = 25; i <= 28; i++)
                    strings[i] = strings[i + 2];
            } else if (strings.length != 29)
                new Throwable(str);

            for (String s : strings) {
                String item[] = s.split(":");
                String v = item[1].replace(" ", "");
                v = v.equals("null") ? "" : v;
                if (encodeMap.get(index).size() == 2 && encodeMap.get(index).containsKey("min") && encodeMap.get(index).containsKey("max")) {
                    if (v.equals("false") || v.equals("true"))
                        res[index] = v.equals("false") ? 0 : 1;
                    else {
                        try {
                            res[index] = Long.parseLong(v);
                        } catch (NumberFormatException e) {
                            res[index] = 0;
                        }
                    }
                } else
                    res[index] = encodeMap.get(index).getOrDefault(v, encodeMap.get(index).size());
                index++;
                if (index >= 29)
                    break;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (index != 29)
            return null;
        return res;
    }

    public void updateValues(long[] features) {
        // column 0 and 2
        long timeStamp = features[0];
        long eventTimeStamp = features[2];

        for (int i = 0; i < 15; i++) {
            crtTimeStampS[i] = crtTimeStampS[i + 1];
            crtEventTimeStampS[i] = crtEventTimeStampS[i + 1];
        }
        crtTimeStampS[15] = timeStamp;
        crtEventTimeStampS[15] = eventTimeStamp;

        // move
        for (int i = 0; i < 29; i++) {
            for (int j = 0; j < 15; j++) {
                values[0][j][i] = values[0][j + 1][i];
            }
        }

        //column 0 and 2
        for (int i = 0; i < 16; i++) {
            long timeStampDelta = crtTimeStampS[i] - crtTimeStampS[0];
            values[0][i][0] = MaxMinScaler(0, timeStampDelta);
            long eventStampDelta = crtEventTimeStampS[i] - crtEventTimeStampS[0];
            values[0][i][2] = MaxMinScaler(2, eventStampDelta);
        }

        features[0] = crtTimeStampS[15] - crtTimeStampS[0];
        features[2] = crtEventTimeStampS[15] - crtEventTimeStampS[0];

        for (int i = 0; i < 29; i++) {
            values[0][15][i] = MaxMinScaler(i, features[i]);
        }
    }

    public float MaxMinScaler(int index,long value) {
        if (max[index] == min[index]) {
            return 0;
        }
        return (value - min[index]) / (max[index] - min[index]);
    }


    public void unbind() {
        model.close();
    }
}
