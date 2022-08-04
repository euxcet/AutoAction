package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.os.Bundle;
import android.util.Log;

import com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.contextactionlibrary.utils.imu.Highpass1C;
import com.hcifuture.contextactionlibrary.utils.imu.Lowpass1C;
import com.hcifuture.contextactionlibrary.utils.imu.MyPeakDetector;
import com.hcifuture.contextactionlibrary.utils.imu.TfClassifier;
import com.hcifuture.contextactionlibrary.utils.imu.Util;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.io.File;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class PocketAction extends BaseAction {

    private String TAG = "PocketAction";

    public static String ACTION = "action.pocket.action";
    public static String ACTION_UPLOAD = "action.pocket.action.upload";
    public static String ACTION_RECOGNIZED = "action.pocket.action.recognized";

    private long SAMPLINGINTERVALNS = 10000000L;
    private long WINDOW_NS = 400000000L;
    private int size = 40;  // WINDOW_NS / SAMPLINGINTERVALNS

    private boolean gotAcc = false;
    private boolean gotGyro = false;
    private long syncTime = 0L;
    private float[] xsAcc = new float[size];
    private float[] ysAcc = new float[size];
    private float[] zsAcc = new float[size];
    private float[] xsGyro = new float[size];
    private float[] ysGyro = new float[size];
    private float[] zsGyro = new float[size];
    private long lastTimestamp = 0L;

    private Highpass1C highpassKeyPositive = new Highpass1C();
    private Lowpass1C lowpassKeyPositive = new Lowpass1C();
    private MyPeakDetector peakDetectorPositive = new MyPeakDetector();
    private boolean wasPositivePeakApproaching = true;
    private long[] doublePocketTimestamps = new long[2];
    private int seqLength;
    private Deque<Long> pocketTimestamps = new ArrayDeque();
    private TfClassifier tflite;

    private ThreadPoolExecutor threadPoolExecutor;

    public PocketAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, actionListener, scheduledExecutorService, futureList);
        init();
        tflite = new TfClassifier(new File(ContextActionContainer.getSavePath() + "pocket.tflite"));
        seqLength = (int)config.getValue("SeqLength");
        threadPoolExecutor = new ThreadPoolExecutor(1, 1, 1000, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(10), Executors.defaultThreadFactory(), new ThreadPoolExecutor.DiscardOldestPolicy());
    }

    private void init() {
        lowpassKeyPositive.setPara(0.2F);
        highpassKeyPositive.setPara(0.2F);
        peakDetectorPositive.setMinNoiseTolerate(0.05f);
        peakDetectorPositive.setWindowSize(40);
    }

    private void reset() {
        gotAcc = false;
        gotGyro = false;
        syncTime = 0L;
        Arrays.fill(xsAcc, 0);
        Arrays.fill(ysAcc, 0);
        Arrays.fill(zsAcc, 0);
        Arrays.fill(xsGyro, 0);
        Arrays.fill(ysGyro, 0);
        Arrays.fill(zsGyro, 0);
        lastTimestamp = 0L;
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Action is already started.");
            return;
        }
        isStarted = true;
        reset();
    }

    @Override
    public synchronized void stop() {
        if (!isStarted) {
            Log.d(TAG, "Action is already stopped");
            return;
        }
        isStarted = false;
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
        if (data.getType() != Sensor.TYPE_GYROSCOPE && data.getType() != Sensor.TYPE_LINEAR_ACCELERATION)
            return;
        Heart.getInstance().newActionAliveEvent(getConfig().getAction(), data.getTimestamp());
        if (data.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            gotAcc = true;
            if (!gotGyro)
                return;
        } else {
            gotGyro = true;
            if (!gotAcc)
                return;
        }
        if (0L == syncTime) {
            syncTime = data.getTimestamp();
            lowpassKeyPositive.init(0.0F);
            highpassKeyPositive.init(0.0F);
        } else {
            if (data.getType() == Sensor.TYPE_LINEAR_ACCELERATION)
                processAccAndKeySignal(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), data.getTimestamp(), SAMPLINGINTERVALNS);
            else
                processGyro(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), SAMPLINGINTERVALNS);
            threadPoolExecutor.execute(() -> {
                recognizePocketML(data.getTimestamp());
            });
        }
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    @Override
    public void onExternalEvent(Bundle bundle) {

    }

    private void processAccAndKeySignal(float x, float y, float z, long t, long samplingInterval) {
        System.arraycopy(xsAcc, 1, xsAcc, 0, size - 1);
        System.arraycopy(ysAcc, 1, ysAcc, 0, size - 1);
        System.arraycopy(zsAcc, 1, zsAcc, 0, size - 1);
        xsAcc[size - 1] = x;
        ysAcc[size - 1] = y;
        zsAcc[size - 1] = z;
        lastTimestamp = t;
        peakDetectorPositive.update(highpassKeyPositive.update(lowpassKeyPositive.update(z)));
    }

    private void processGyro(float x, float y, float z, long samplingInterval) {
        System.arraycopy(xsGyro, 1, xsGyro, 0, size - 1);
        System.arraycopy(ysGyro, 1, ysGyro, 0, size - 1);
        System.arraycopy(zsGyro, 1, zsGyro, 0, size - 1);
        xsGyro[size - 1] = x;
        ysGyro[size - 1] = y;
        zsGyro[size - 1] = z;
    }

    private ArrayList<Float> getInput(int idx) {
        ArrayList<Float> res = new ArrayList<>();
        addFeatureData(xsAcc, idx, 1, res);
        addFeatureData(ysAcc, idx, 1, res);
        addFeatureData(zsAcc, idx, 1, res);
        addFeatureData(xsGyro, idx, 1, res);
        addFeatureData(ysGyro, idx, 1, res);
        addFeatureData(zsGyro, idx, 1, res);
        return res;
    }

    private void addFeatureData(float[] list, int startIdx, int scale, List<Float> res) {
        for (int i = 0; i < seqLength; i++) {
            if (i + startIdx >= size)
                res.add(0.0F);
            else
                res.add(scale * list[i + startIdx]);
        }
    }

    private synchronized int checkDoublePocketTiming(long timestamp) {
        Iterator iter = pocketTimestamps.iterator();
        while (iter.hasNext()) {
            if (timestamp - (Long) iter.next() > 500000000L) {
                iter.remove();
            }
        }
        int res = 0;
        if (!pocketTimestamps.isEmpty()) {
            iter = pocketTimestamps.iterator();
            while (true) {
                if (!iter.hasNext()) {
                    res = 1;
                    break;
                }
                doublePocketTimestamps[1] = (Long)pocketTimestamps.getLast();
                doublePocketTimestamps[0] = (Long)iter.next();
                if (doublePocketTimestamps[1] - doublePocketTimestamps[0] > 100000000L) {
                    pocketTimestamps.clear();
                    res = 2;
                    break;
                }
            }
        }
        return res;
    }

    public long getFirstPocketTimestamp() {
        return doublePocketTimestamps[0];
    }

    public long getSecondPocketTimestamp() {
        return doublePocketTimestamps[1];
    }

    public void recognizePocketML(long timestamp) {
        int result = 0;
        int peakIdxPositive = peakDetectorPositive.getIdMajorPeak();
        if (peakIdxPositive == 32) {
            wasPositivePeakApproaching = true;
        }
        int idxPositive = peakIdxPositive - 15;
        if (idxPositive >= 0) {
            if (idxPositive + seqLength < size && wasPositivePeakApproaching && peakIdxPositive <= 30) {
                int tmp = Util.getMaxId(tflite.predict(getInput(idxPositive), 2, true).get(0));
                if (tmp == 1) {
                    wasPositivePeakApproaching = false;
                    peakDetectorPositive.reset();
                    result = 1;
                }
            }
        }
        else
            wasPositivePeakApproaching = false;
        if (result == 1) {
            pocketTimestamps.addLast(timestamp);
            int count = checkDoublePocketTiming(lastTimestamp);
            if (count == 2) {
                if (actionListener != null) {
                    for (ActionListener listener : actionListener) {
                        ActionResult actionResult = new ActionResult(ACTION_RECOGNIZED);
                        actionResult.getExtras().putLongArray("timestamp", new long[]{getFirstPocketTimestamp(), getSecondPocketTimestamp()});
                        listener.onAction(actionResult);
                        actionResult.setAction(ACTION);
                        listener.onAction(actionResult);
                        actionResult.setAction(ACTION_UPLOAD);
                        actionResult.setReason("Triggered");
                        listener.onAction(actionResult);
                        // listener.onActionSave(actionResult);
                    }
                }
            }
        }
    }

    @Override
    public synchronized void getAction() {
        /*
        if (!isStarted)
            return;
        int count = checkDoublePocketTiming(lastTimestamp);
        if (count == 2) {
            if (actionListener != null) {
                for (ActionListener listener : actionListener) {
                    ActionResult actionResult = new ActionResult(ACTION_RECOGNIZED);
//                    actionResult.setTimestamp(getFirstPocketTimestamp() + ":" + getSecondPocketTimestamp());
                    actionResult.getExtras().putLongArray("timestamp", new long[]{getFirstPocketTimestamp(), getSecondPocketTimestamp()});
                    listener.onAction(actionResult);
                    actionResult.setAction(ACTION);
                    listener.onAction(actionResult);
                    actionResult.setAction(ACTION_UPLOAD);
                    actionResult.setReason("Triggered");
                    listener.onAction(actionResult);
                    // listener.onActionSave(actionResult);
                }
            }
        }
         */
    }

    @Override
    public String getName() {
        return "PocketAction";
    }
}
