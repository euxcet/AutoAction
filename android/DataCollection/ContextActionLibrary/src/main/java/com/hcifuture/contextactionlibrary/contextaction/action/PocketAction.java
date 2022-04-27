package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.util.Log;

import com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
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
import java.util.Collections;
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
    private int size = 40;

    private boolean gotAcc = false;
    private boolean gotGyro = false;
    private long syncTime = 0L;
    private List<Float> xsAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> ysAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> zsAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> xsGyro = Collections.synchronizedList(new ArrayList<>());
    private List<Float> ysGyro = Collections.synchronizedList(new ArrayList<>());
    private List<Float> zsGyro = Collections.synchronizedList(new ArrayList<>());
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
        xsAcc.clear();
        ysAcc.clear();
        zsAcc.clear();
        xsGyro.clear();
        ysGyro.clear();
        zsGyro.clear();
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

    private void processAccAndKeySignal(float x, float y, float z, long t, long samplingInterval) {
        xsAcc.add(x);
        ysAcc.add(y);
        zsAcc.add(z);
        lastTimestamp = t;

        while(xsAcc.size() > size) {
            xsAcc.remove(0);
            ysAcc.remove(0);
            zsAcc.remove(0);
        }

        peakDetectorPositive.update(highpassKeyPositive.update(lowpassKeyPositive.update(z)));
    }

    private void processGyro(float x, float y, float z, long samplingInterval) {
        xsGyro.add(x);
        ysGyro.add(y);
        zsGyro.add(z);

        while(xsGyro.size() > size) {
            xsGyro.remove(0);
            ysGyro.remove(0);
            zsGyro.remove(0);
        }
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

    private void addFeatureData(List<Float> list, int startIdx, int scale, List<Float> res) {
        synchronized (list) {
            for (int i = 0; i < seqLength; i++) {
                if (i + startIdx >= list.size())
                    res.add(0.0F);
                else
                    res.add(scale * list.get(i + startIdx));
            }
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
}
