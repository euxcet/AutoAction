package com.hcifuture.datacollection.inference;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Network;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Pair;
import android.widget.Toast;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.inference.mnn.MNNForwardType;
import com.hcifuture.datacollection.inference.mnn.MNNImageProcess;
import com.hcifuture.datacollection.inference.mnn.MNNNetInstance;
import com.hcifuture.datacollection.inference.mnn.MNNNetNative;
import com.hcifuture.datacollection.inference.utils.Common;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.model.Response;

import org.checkerframework.checker.units.qual.A;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Inferencer {
    private static volatile Inferencer instance;

    private Context mContext;

    private final List<String> ASSETS_FILES = Arrays.asList("best.mnn", "action.mnn", "siamese_export.mnn", "label.txt");

    private List<String> mActionLabels;

    private HandlerThread mThread;
    private Handler mHandler;

    private final MNNNetInstance.Config mConfig = new MNNNetInstance.Config();

    private List<String> mModelPaths = new ArrayList<>();
    private List<MNNNetInstance> mNetInstances = new ArrayList<>();
    private List<MNNNetInstance.Session> mSessions = new ArrayList<>();
    private List<MNNNetInstance.Session.Tensor> mInputTensors = new ArrayList<>();

    private final Lock lock = new ReentrantLock();
    private AtomicBoolean isStarted = new AtomicBoolean(false);
    private AtomicBoolean isDownloading = new AtomicBoolean(false);

    Matrix imgData;

    private String currentModelId;

    private Inferencer() {
        imgData = new Matrix();
    }

    public String getCurrentModelId() {
        return currentModelId;
    }

    public List<String> getActionLabels() {
        return mActionLabels;
    }

    private List<String> getLabels() {
        List<String> labels = new ArrayList<>();
        File labelFile = new File(getPath("label.txt"));
        FileInputStream fis = null;
        try {
            fis = new FileInputStream(labelFile);
            InputStreamReader reader = new InputStreamReader(fis, "UTF-8");
            BufferedReader br = new BufferedReader(reader);
            String line;
            while ((line = br.readLine()) != null) {
                if (line.length() > 0) {
                    labels.add(line);
                }
            }
            fis.close();
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return labels;
    }

    private void prepareModel(Context context) {
        try {
            for (String name: ASSETS_FILES) {
                Common.copyAssetResource2File(context, name, getPath(name));
            }
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    private String getPath(String name) {
        return mContext.getCacheDir() + name;
    }

    public static Inferencer getInstance() {
        if (instance == null) {
            synchronized (Inferencer.class) {
                if (instance == null) {
                    instance = new Inferencer();
                }
            }
        }
        return instance;
    }

    public void start(Context context) {
        mContext = context;
        isStarted.set(false);
        isDownloading.set(false);
        prepareModel(context);
        mActionLabels = getLabels();
        mConfig.numThread = 4;
        mConfig.forwardType = MNNForwardType.FORWARD_CPU.type;
        mThread = new HandlerThread("MNNNet");
        mThread.start();
        mHandler = new Handler(mThread.getLooper());
        mHandler.post(() -> {
            isStarted.set(false);
            for (String name: ASSETS_FILES) {
                if (name.endsWith(".mnn")) {
                    prepareNet(name);
                }
            }
            isStarted.set(true);
        });
        currentModelId = "Assets";
    }

    public void useModel(Context context, String trainId) {
        if (isDownloading.get()) {
            return;
        }
        isDownloading.set(true);
        mHandler.post(() -> {
            Toast.makeText(context, "Downloading model [" + trainId + "]", Toast.LENGTH_LONG).show();
        });
        String path = context.getCacheDir() + "action.mnn";
        NetworkUtils.downloadTrainLabel(trainId, new FileCallback() {
            @Override
            public void onSuccess(Response<File> response) {
                File file = response.body();
                FileUtils.copy(file, new File(path));
                file.delete();
                NetworkUtils.downloadTrainMNNModel(trainId, new FileCallback() {
                    @Override
                    public void onSuccess(Response<File> response) {
                        File file = response.body();
                        FileUtils.copy(file, new File(path));
                        file.delete();
                        prepareNet(path);
                        mHandler.post(() -> {
                            Toast.makeText(context, "Use model [" + trainId + "]", Toast.LENGTH_LONG).show();
                        });
                        currentModelId = trainId;
                        isDownloading.set(false);
                    }
                });
            }
        });
    }

    private void prepareNet(String name) {
        String path = getPath(name);
        try {
            lock.lock();
            int pos = -1;
            for(int i = 0; i < mModelPaths.size(); i++) {
                if (mModelPaths.get(i).equals(path)) {
                    pos = i;
                    break;
                }
            }
            if (pos == -1) { // create a new model
                MNNNetInstance netInstance = MNNNetInstance.createFromFile(path);
                MNNNetInstance.Session session = netInstance.createSession(mConfig);
                MNNNetInstance.Session.Tensor inputTensor = session.getInput(null);
                mModelPaths.add(path);
                mNetInstances.add(netInstance);
                mSessions.add(session);
                mInputTensors.add(inputTensor);
            } else {
                if (mSessions.get(pos) != null) {
                    mSessions.get(pos).release();
                }
                if (mNetInstances.get(pos) != null) {
                    mNetInstances.get(pos).release();
                }

                MNNNetInstance netInstance = MNNNetInstance.createFromFile(path);
                MNNNetInstance.Session session = netInstance.createSession(mConfig);
                MNNNetInstance.Session.Tensor inputTensor = session.getInput(null);
                mNetInstances.set(pos, netInstance);
                mSessions.set(pos, session);
                mInputTensors.set(pos, inputTensor);
            }
        } finally {
            lock.unlock();
        }
    }

    private String getLabelById(int id) {
        if (id < 0 || id >= mActionLabels.size()) {
            return "[ERROR]INDEX_OUT_OF_BOUNDS";
        }
        return mActionLabels.get(id);
    }

    public InferenceResult inferenceAction(String name, float[] data) {
        String path = getPath(name);
        int result = -1;
        float max_value = -1.0f;
        if (isStarted.get()) {
            try {
                lock.lock();
                for (int i = 0; i < mModelPaths.size(); i++) {
                    if (mModelPaths.get(i).equals(path)) {
                        mInputTensors.get(i).setInputFloatData(data);
                        mSessions.get(i).run();
                        MNNNetInstance.Session.Tensor outputTensor = mSessions.get(i).getOutput(null);
                        float[] output = outputTensor.getFloatData();
                        for (int p = 0; p < output.length; p++) {
                            if (output[p] > max_value) {
                                max_value = output[p];
                                result = p;
                            }
                        }

                    }
                }
            } finally {
                lock.unlock();
            }
        }
        return new InferenceResult(result, getLabelById(result), max_value);
    }

    public float[] inferenceImage(String name, Bitmap bitmap) {
        String path = getPath(name);
        float[] result = null;
        if (isStarted.get()) {
            try {
                lock.lock();
                for (int i = 0; i < mModelPaths.size(); i++) {
                    if (mModelPaths.get(i).equals(path)) {
                        MNNImageProcess.Config dataConfig = new MNNImageProcess.Config();
                        dataConfig.mean = new float[]{128.0f, 128.0f, 128.0f};
                        dataConfig.normal= new float[]{0.0078125f, 0.0078125f, 0.0078125f};
//                        dataConfig.mean = new float[]{124.16f, 116.736f, 103.936f};
//                        dataConfig.normal= new float[]{56.624f, 57.344f, 57.6f};
                        dataConfig.dest = MNNImageProcess.Format.RGB;

                        MNNImageProcess.convertBitmap(bitmap, mInputTensors.get(i), dataConfig, imgData);
                        float[] x = mInputTensors.get(i).getFloatData();
                        Log.e("TEST", "input data " + x.length + " " + x[0] + " " + x[1] + " " + x[2]);
                        mSessions.get(i).run();
                        MNNNetInstance.Session.Tensor outputTensor = mSessions.get(i).getOutput(null);
                        result = outputTensor.getFloatData();
                    }
                }
            } finally {
                lock.unlock();
            }
        }
        return result;
    }

}
