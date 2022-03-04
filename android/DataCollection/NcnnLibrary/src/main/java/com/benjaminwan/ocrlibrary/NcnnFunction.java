package com.benjaminwan.ocrlibrary;

import android.content.Context;
import android.content.res.AssetManager;

public class NcnnFunction {
    private int width;
    private int height;
    private int channel;
    public NcnnFunction(Context context, String paramPath, String binPath, int numThread, int width, int height, int channel, int class_num) {
        System.loadLibrary("OcrLite");
        init(paramPath, binPath, numThread, width, height, channel, class_num);
        // init(context.getAssets(), 4);
    }

    public native boolean initByAsset(AssetManager assetManager, int numThread);
    public native boolean init(String paramPath, String binPath, int numThread, int width, int height, int channel, int class_num);
    public native void print();
    public native int actionDetect(float[] data);
}