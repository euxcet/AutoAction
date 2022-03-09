package com.example.contextactionlibrary.model;

import android.content.Context;

import com.example.ncnnlibrary.NcnnFunction;


public class NcnnInstance {
    private volatile static NcnnInstance mInstance;
    private NcnnFunction ncnnFunction;

    private NcnnInstance(Context context, String paramPath, String binPath, int numThread, int inputWidth, int inputHeight, int inputChannel, int classNum) {
        ncnnFunction = new NcnnFunction(context, paramPath, binPath, numThread, inputWidth, inputHeight, inputChannel, classNum);
    }

    public int actionDetect(float[] data) {
        return ncnnFunction.actionDetect(data);
    }

    public static void init(Context context, String paramPath, String binPath, int numThread, int inputWidth, int inputHeight, int inputChannel, int classNum) {
        mInstance = new NcnnInstance(context, paramPath, binPath, numThread, inputWidth, inputHeight, inputChannel, classNum);
    }

    public static NcnnInstance getInstance() {
        return mInstance;
    }
}
