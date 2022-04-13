package com.hcifuture.contextactionlibrary.utils.imu;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import org.tensorflow.lite.Interpreter;

public class TfClassifier {
    Interpreter interpreter = null;

    public TfClassifier(File var2) {
        try {
            Interpreter var10 = new Interpreter(var2);
            this.interpreter = var10;
            StringBuilder var11 = new StringBuilder();
            var11.append("tflite file loaded: ");
            var11.append(var2);
        } catch (Exception var8) {
            StringBuilder var3 = new StringBuilder();
            var3.append("load tflite file error: ");
            var3.append(var2);
            StringBuilder var12 = new StringBuilder();
            var12.append("tflite file:");
            var12.append(var8.toString());
        }

    }

    public TfClassifier(AssetManager var1, String var2) {
        try {
            AssetFileDescriptor var9 = var1.openFd(var2);
            FileInputStream var13 = new FileInputStream(var9.getFileDescriptor());
            FileChannel var14 = var13.getChannel();
            long var4 = var9.getStartOffset();
            long var6 = var9.getDeclaredLength();
            MappedByteBuffer var15 = var14.map(FileChannel.MapMode.READ_ONLY, var4, var6);
            Interpreter var10 = new Interpreter(var15);
            this.interpreter = var10;
            StringBuilder var11 = new StringBuilder();
            var11.append("tflite file loaded: ");
            var11.append(var2);
        } catch (Exception var8) {
            StringBuilder var3 = new StringBuilder();
            var3.append("load tflite file error: ");
            var3.append(var2);
            StringBuilder var12 = new StringBuilder();
            var12.append("tflite file:");
            var12.append(var8.toString());
        }

    }

    public ArrayList<ArrayList<Float>> predict(ArrayList<Float> var1, int var2) {
        ArrayList<ArrayList<Float>> res = new ArrayList<>();
        if (this.interpreter != null) {
            float[] var3 = new float[var1.size()];

            int var4;
            for(var4 = 0; var4 < var1.size(); ++var4) {
                var3[var4] = (Float)var1.get(var4);
            }

            HashMap var6 = new HashMap();
            var6.put(0, new float[1][var2]);
            this.interpreter.runForMultipleInputsOutputs(new Object[]{var3}, var6);
            float[][] var5 = (float[][])var6.get(0);
            ArrayList var7 = new ArrayList();

            for(var4 = 0; var4 < var2; ++var4) {
                var7.add(var5[0][var4]);
            }

            res.add(var7);
        }

        return res;
    }

    public ArrayList<ArrayList<Float>> predict(ArrayList<Float> var1, int var2, boolean isPocket) {
        ArrayList<ArrayList<Float>> res = new ArrayList<>();
        if (this.interpreter != null) {
            float[][][][] var3 = new float[1][1][var1.size()][1];

            int var4;
            for(var4 = 0; var4 < var1.size(); ++var4) {
                var3[0][0][var4][0] = (Float)var1.get(var4);
            }

            HashMap var6 = new HashMap();
            var6.put(0, new float[1][var2]);
            this.interpreter.runForMultipleInputsOutputs(new Object[]{var3}, var6);
            float[][] var5 = (float[][])var6.get(0);
            ArrayList var7 = new ArrayList();

            for(var4 = 0; var4 < var2; ++var4) {
                var7.add(var5[0][var4]);
            }

            res.add(var7);
        }

        return res;
    }
}
