package com.hcifuture.datacollection.inference;

import android.content.Context;
import android.net.Network;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.widget.Toast;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.inference.mnn.MNNForwardType;
import com.hcifuture.datacollection.inference.mnn.MNNNetInstance;
import com.hcifuture.datacollection.inference.mnn.MNNNetNative;
import com.hcifuture.datacollection.inference.utils.Common;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Inferencer {
    private static volatile Inferencer instance;
    private final String mMobileNetName = "mobilenet.mnn";
    private final String mModelFileName = "best.mnn";
    private final String mLabelFileName = "label.txt";
    private String mModelPath;
    private String mLabelPath;
    private String mMobileNetPath;

    private List<String> label;

    private HandlerThread mThread;
    private Handler mHandler;

    /*
    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
     */
    private final MNNNetInstance.Config mConfig = new MNNNetInstance.Config();

    private List<String> mModelPaths;
    private List<MNNNetInstance> mNetInstances;
    private List<MNNNetInstance.Session> mSessions;
    private List<MNNNetInstance.Session.Tensor> mInputTensors;

    private final Lock lock = new ReentrantLock();
    private AtomicBoolean isStarted = new AtomicBoolean(false);
    private AtomicBoolean isDownloading = new AtomicBoolean(false);

    private String currentModelId;

    private Inferencer() {
    }

    public String getCurrentModelId() {
        return currentModelId;
    }

    public List<String> getLabel() {
        return label;
    }

    public void start(Context context) {
        isStarted.set(false);
        isDownloading.set(false);
        prepareModel(context);
        mConfig.numThread = 4;
        mConfig.forwardType = MNNForwardType.FORWARD_CPU.type;
        mThread = new HandlerThread("MNNNet");
        mThread.start();
        mHandler = new Handler(mThread.getLooper());
        mHandler.post(() -> {
            isStarted.set(false);
            prepareNet(mModelPath);
            prepareNet(mMobileNetPath);
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
        mModelPath = context.getCacheDir() + "best.mnn";
        mLabelPath = context.getCacheDir() + "label.txt";
        NetworkUtils.downloadTrainLabel(trainId, new FileCallback() {
            @Override
            public void onSuccess(Response<File> response) {
                File file = response.body();
                FileUtils.copy(file, new File(mLabelPath));
                file.delete();
                NetworkUtils.downloadTrainMNNModel(trainId, new FileCallback() {
                    @Override
                    public void onSuccess(Response<File> response) {
                        File file = response.body();
                        FileUtils.copy(file, new File(mModelPath));
                        file.delete();
                        prepareNet(mModelPath);
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

    private void prepareNet(String path) {
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

    public int inference(String path, float[] data) {
        if (isStarted.get()) {
            try {
                lock.lock();
                for (int i = 0; i < mModelPaths.size(); i++) {
                    if (mModelPaths.get(i).equals(path)) {
                        mInputTensor.setInputFloatData(data);
                        mSession.run();
                        MNNNetInstance.Session.Tensor outputTensor = mSession.getOutput(null);
                        float[] result = outputTensor.getFloatData();
                    }
                }
            } finally {
                lock.unlock();
            }
        }
        return -1;
    }


    private void prepareModel(Context context) {
        mModelPath = context.getCacheDir() + "best.mnn";
        mLabelPath = context.getCacheDir() + "label.txt";
        mMobileNetPath = context.getCacheDir() + "mobilenet.mnn";
        try {
            Common.copyAssetResource2File(context, mModelFileName, mModelPath);
            Common.copyAssetResource2File(context, mLabelFileName, mLabelPath);
            Common.copyAssetResource2File(context, mMobileNetName, mMobileNetPath);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
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
}
