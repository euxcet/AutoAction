package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.util.Log;

import com.hcifuture.contextactionlibrary.BuildConfig;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PocketAction extends BaseAction {

    private String TAG = "PocketAction";

    private long SAMPLINGINTERVALNS = 10000000L;
    private long WINDOW_NS = 400000000L;

    private boolean gotAcc = false;
    private boolean gotGyro = false;
    private long syncTime = 0L;
    private List<Float> xsAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> ysAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> zsAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> xsGyro = Collections.synchronizedList(new ArrayList<>());
    private List<Float> ysGyro = Collections.synchronizedList(new ArrayList<>());
    private List<Float> zsGyro = Collections.synchronizedList(new ArrayList<>());
    private List<Long> timestamps = Collections.synchronizedList(new ArrayList<>());

    private Highpass1C highpassKeyPositive = new Highpass1C();
    private Lowpass1C lowpassKeyPositive = new Lowpass1C();
    private MyPeakDetector peakDetectorPositive = new MyPeakDetector();
    private boolean wasPositivePeakApproaching = true;
    private long[] doublePocketTimestamps = new long[2];
    private int result;
    private int seqLength;
    private List<Long> pocketTimestamps = Collections.synchronizedList(new ArrayList<>());
    private TfClassifier tflite;

    public PocketAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener) {
        super(context, config, requestListener, actionListener);
        init();
        tflite = new TfClassifier(new File(BuildConfig.SAVE_PATH + "pocket.tflite"));
        seqLength = (int)config.getValue("SeqLength");
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
        timestamps.clear();
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
    public synchronized void onIMUSensorChanged(SensorEvent event) {
        if (event.sensor.getType() != Sensor.TYPE_GYROSCOPE && event.sensor.getType() != Sensor.TYPE_LINEAR_ACCELERATION)
            return;
        result = 0;
        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            gotAcc = true;
            if (!gotGyro)
                return;
        } else {
            gotGyro = true;
            if (!gotAcc)
                return;
        }
        if (0L == syncTime) {
            syncTime = event.timestamp;
            lowpassKeyPositive.init(0.0F);
            highpassKeyPositive.init(0.0F);
        } else {
            if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION)
                processAccAndKeySignal(event.values[0], event.values[1], event.values[2], event.timestamp, SAMPLINGINTERVALNS);
            else
                processGyro(event.values[0], event.values[1], event.values[2], SAMPLINGINTERVALNS);
            recognizePocketML();
            if (result == 1) {
                pocketTimestamps.add(event.timestamp);
            }
        }
    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {

    }

    private void processAccAndKeySignal(float x, float y, float z, long t, long samplingInterval) {
        xsAcc.add(x);
        ysAcc.add(y);
        zsAcc.add(z);
        timestamps.add(t);
        int size = (int)(WINDOW_NS / samplingInterval);

        while(xsAcc.size() > size) {
            xsAcc.remove(0);
            ysAcc.remove(0);
            zsAcc.remove(0);
            timestamps.remove(0);
        }

        peakDetectorPositive.update(highpassKeyPositive.update(lowpassKeyPositive.update(z)));
    }

    private void processGyro(float x, float y, float z, long samplingInterval) {
        xsGyro.add(x);
        ysGyro.add(y);
        zsGyro.add(z);
        int size = (int)(WINDOW_NS / samplingInterval);

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

    private int checkDoublePocketTiming(long timestamp) {
        int res = 0;
        synchronized (pocketTimestamps) {
            // remove old timestamps
            int idx = 0;
            for (; idx < pocketTimestamps.size(); idx++) {
                if (timestamp - pocketTimestamps.get(idx) <= 500000000L)
                    break;
            }
            pocketTimestamps = pocketTimestamps.subList(idx, pocketTimestamps.size());

            if (pocketTimestamps.isEmpty())
                res = 0;
            else {
                if (pocketTimestamps.size() == 1)
                    res = 1;
                doublePocketTimestamps[1] = pocketTimestamps.get(pocketTimestamps.size() - 1);
                doublePocketTimestamps[0] = pocketTimestamps.get(0);
                if (doublePocketTimestamps[1] - doublePocketTimestamps[0] > 100000000L) {
                    pocketTimestamps.clear();
                    res = 2;
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

    public void recognizePocketML() {
        int peakIdxPositive = peakDetectorPositive.getIdMajorPeak();
        if (peakIdxPositive == 32) {
            wasPositivePeakApproaching = true;
        }
        int idxPositive = peakIdxPositive - 15;
        if (idxPositive >= 0) {
            if (idxPositive + seqLength < zsAcc.size() && wasPositivePeakApproaching && peakIdxPositive <= 30) {
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
    }

    @Override
    public synchronized void getAction() {
        if (!isStarted)
            return;
        long timestamp = timestamps.get(seqLength);
        int count = checkDoublePocketTiming(timestamp);
        if (count == 2) {
            if (actionListener != null) {
                for (ActionListener listener : actionListener) {
                    ActionResult actionResult = new ActionResult("Pocket");
                    actionResult.setTimestamp(getFirstPocketTimestamp() + ":" + getSecondPocketTimestamp());
                    actionResult.setReason("Triggered");
                    listener.onAction(actionResult);
                    listener.onActionSave(actionResult);
                }
            }
        }
    }
}