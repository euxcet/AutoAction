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
    private final String mModelFileName = "best.mnn";
    private final String mLabelFileName = "label.txt";
    private String mModelPath;
    private String mLabelPath;

    private List<String> label;

    private HandlerThread mThread;
    private Handler mHandler;

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    private final MNNNetInstance.Config mConfig = new MNNNetInstance.Config();

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
        mHandler.post(() -> prepareNet());
        currentModelId = "Assets";
    }

    public void useModel(Context context, String trainId) {
        if (isDownloading.get()) {
            return;
        }
        isDownloading.set(true);
        Log.e("MNN", "use model " + trainId);
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
                        prepareNet();
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

    private void prepareNet() {
        try {
            lock.lock();
            isStarted.set(false);
            if (mSession != null) {
                mSession.release();
                mSession = null;
            }
            if (mNetInstance != null) {
                mNetInstance.release();
                mNetInstance = null;
            }

            label = FileUtils.readLines(mLabelPath);
            Log.e("MNN", label + " ");

            mNetInstance = MNNNetInstance.createFromFile(mModelPath);
            mSession = mNetInstance.createSession(mConfig);
            mInputTensor = mSession.getInput(null);
            isStarted.set(true);
        } finally {
            lock.unlock();
        }
    }

    public int inference(float[] data) {
        if (isStarted.get()) {
            try {
                lock.lock();
                mInputTensor.setInputFloatData(data);
                mSession.run();
                MNNNetInstance.Session.Tensor outputTensor = mSession.getOutput(null);
                float[] result = outputTensor.getFloatData();
                Log.e("MNN", result.length + "   " + result[0] + " " + result[1] + " " + result[2]);
            } finally {
                lock.unlock();
            }
        }
        return -1;
    }


    private void prepareModel(Context context) {
        mModelPath = context.getCacheDir() + "best.mnn";
        mLabelPath = context.getCacheDir() + "label.txt";
        try {
            Common.copyAssetResource2File(context, mModelFileName, mModelPath);
            Common.copyAssetResource2File(context, mLabelFileName, mLabelPath);
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
