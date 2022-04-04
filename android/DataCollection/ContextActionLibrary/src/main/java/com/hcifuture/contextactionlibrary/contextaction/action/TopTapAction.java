package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.util.Log;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.hcifuture.contextactionlibrary.contextaction.action.tapfilter.CombinedFilter;
import com.hcifuture.contextactionlibrary.contextaction.action.tapfilter.HorizontalFilter;
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

import kotlin.jvm.Synchronized;

public class TopTapAction extends BaseAction {

    private String TAG = "TopTapAction";

    private long SAMPLINGINTERVALNS = 10000000L;
    private long WINDOW_NS = 400000000L;

    private boolean gotAcc = false;
    private boolean gotGyro = false;
    private long syncTime = 0L;
    private List<Float> xsAcc = new ArrayList<>();
    private List<Float> ysAcc = new ArrayList<>();
    private List<Float> zsAcc = new ArrayList<>();
    private List<Float> xsGyro = new ArrayList<>();
    private List<Float> ysGyro = new ArrayList<>();
    private List<Float> zsGyro = new ArrayList<>();
    private List<Long> timestamps = new ArrayList<>();

    private Highpass1C highpassKeyPositive = new Highpass1C();
    private Lowpass1C lowpassKeyPositive = new Lowpass1C();
    private Highpass1C highpassKeyNegative = new Highpass1C();
    private Lowpass1C lowpassKeyNegative = new Lowpass1C();
    private MyPeakDetector peakDetectorPositive = new MyPeakDetector();
    private MyPeakDetector peakDetectorNegative = new MyPeakDetector();
    private boolean wasPositivePeakApproaching = true;
    private boolean wasNegativePeakApproaching = true;
    private long[] doubleBackTapTimestamps = new long[2];
    private long[] doubleTopTapTimestamps = new long[2];
    private int result;
    private int seqLength;
    private List<Long> backTapTimestamps = new ArrayList();
    private List<Long> topTapTimestamps = new ArrayList();
    private TfClassifier tflite;

    // filter related
    private HorizontalFilter horizontalFilter = new HorizontalFilter();


    public TopTapAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener) {
        super(context, config, requestListener, actionListener);
        init();
        // tflite = new TfClassifier(mContext.getAssets(), "combined.tflite");
        tflite = new TfClassifier(new File(BuildConfig.SAVE_PATH + "combined.tflite"));
        seqLength = (int)config.getValue("SeqLength");
    }

    private void init() {
        lowpassKeyPositive.setPara(0.2F);
        highpassKeyPositive.setPara(0.2F);
        lowpassKeyNegative.setPara(0.2F);
        highpassKeyNegative.setPara(0.2F);
        peakDetectorPositive.setMinNoiseTolerate(0.05f);
        peakDetectorPositive.setWindowSize(40);
        peakDetectorNegative.setMinNoiseTolerate(0.05f);
        peakDetectorNegative.setWindowSize(40);
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
        // just for horizontal / static cases' record && upload
        horizontalFilter.onSensorChanged(event);
        if (horizontalFilter.passWithDelay(event.timestamp) == -1) {
            ActionResult actionResult = new ActionResult("TopTap");
            actionResult.setReason("Static");
            for (ActionListener listener : actionListener) {
                listener.onActionSave(actionResult);
            }
        }

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
            lowpassKeyNegative.init(0.0F);
            highpassKeyNegative.init(0.0F);
        } else {
            if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION)
                processAccAndKeySignal(event.values[0], event.values[1], event.values[2], event.timestamp, SAMPLINGINTERVALNS);
            else
                processGyro(event.values[0], event.values[1], event.values[2], SAMPLINGINTERVALNS);
            recognizeTapML();
            if (result == 1) {
                backTapTimestamps.add(event.timestamp);
            }
            else if (result == 2) {
                topTapTimestamps.add(event.timestamp);
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
        peakDetectorNegative.update(-highpassKeyNegative.update(lowpassKeyNegative.update(y)));
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
        for (int i = 0; i < seqLength; i++) {
            if (i + startIdx >= list.size())
                res.add(0.0F);
            else
                res.add(scale * list.get(i + startIdx));
        }
    }

    private int checkDoubleBackTapTiming(long timestamp) {
        // remove old timestamps
        int idx = 0;
        for (; idx < backTapTimestamps.size(); idx++) {
            if (timestamp - backTapTimestamps.get(idx) <= 500000000L)
                break;
        }
        backTapTimestamps = backTapTimestamps.subList(idx, backTapTimestamps.size());

        if (backTapTimestamps.isEmpty())
            return 0;
        else {
            if (backTapTimestamps.size() == 1)
                return 1;
            doubleBackTapTimestamps[1] = backTapTimestamps.get(backTapTimestamps.size() - 1);
            doubleBackTapTimestamps[0] = backTapTimestamps.get(0);
            if (doubleBackTapTimestamps[1] - doubleBackTapTimestamps[0] > 100000000L) {
                backTapTimestamps.clear();
                return 2;
            }
        }
        return 0;
    }

    private int checkDoubleTopTapTiming(long timestamp) {
        // remove old timestamps
        int idx = 0;
        for (; idx < topTapTimestamps.size(); idx++) {
            if (timestamp - topTapTimestamps.get(idx) <= 500000000L)
                break;
        }
        topTapTimestamps = topTapTimestamps.subList(idx, topTapTimestamps.size());

        if (topTapTimestamps.isEmpty())
            return 0;
        else {
            if (topTapTimestamps.size() == 1)
                return 1;
            doubleTopTapTimestamps[1] = topTapTimestamps.get(topTapTimestamps.size() - 1);
            doubleTopTapTimestamps[0] = topTapTimestamps.get(0);
            if (doubleTopTapTimestamps[1] - doubleTopTapTimestamps[0] > 100000000L) {
                topTapTimestamps.clear();
                return 2;
            }
        }
        return 0;
    }

    public float getFirstBackTapTimestamp() {
        return (float) (doubleBackTapTimestamps[0] / 1000000);
    }

    public float getSecondBackTapTimestamp() {
        return (float) (doubleBackTapTimestamps[1] / 1000000);
    }

    public float getFirstTopTapTimestamp() {
        return (float) (doubleTopTapTimestamps[0] / 1000000);
    }

    public float getSecondTopTapTimestamp() {
        return (float) (doubleTopTapTimestamps[1] / 1000000);
    }

    public void recognizeTapML() {
        // for taptap
        int peakIdxPositive = peakDetectorPositive.getIdMajorPeak();
        if (peakIdxPositive == 32) {
            wasPositivePeakApproaching = true;
        }
        int idxPositive = peakIdxPositive - 15;
        if (idxPositive >= 0) {
            if (idxPositive + seqLength < zsAcc.size() && wasPositivePeakApproaching && peakIdxPositive <= 30) {
                int tmp = Util.getMaxId(tflite.predict(getInput(idxPositive), 3).get(0));
                if (tmp == 1) {
                    wasPositivePeakApproaching = false;
                    peakDetectorPositive.reset();
                    result = 1;
                }
                else if (tmp == 2) {
                    wasNegativePeakApproaching = false;
                    peakDetectorNegative.reset();
                    result = 2;
                }
            }
        }
        else
            wasPositivePeakApproaching = false;
        // for toptap
        int peakIdxNegative = peakDetectorNegative.getIdMajorPeak();
        if (peakIdxNegative == 32) {
            wasNegativePeakApproaching = true;
        }
        int idxNegative = peakIdxNegative - 15;
        if (idxNegative >= 0) {
            if (idxNegative + seqLength < zsAcc.size() && wasNegativePeakApproaching && peakIdxNegative <= 30) {
                int tmp = Util.getMaxId(tflite.predict(getInput(idxNegative), 3).get(0));
                if (tmp == 1) {
                    wasPositivePeakApproaching = false;
                    peakDetectorPositive.reset();
                    result = 1;
                }
                else if (tmp == 2) {
                    wasNegativePeakApproaching = false;
                    peakDetectorNegative.reset();
                    result = 2;
                }
            }
        }
        else
            wasNegativePeakApproaching = false;
    }

    @Override
    public synchronized void getAction() {
        if (!isStarted)
            return;
        long timestamp = timestamps.get(seqLength);
        int count1 = checkDoubleBackTapTiming(timestamp);
        if (count1 == 2) {
            if (actionListener != null) {
                for (ActionListener listener: actionListener) {
                    ActionResult actionResult = new ActionResult("TapTapConfirmed");
                    actionResult.setTimestamp(getFirstBackTapTimestamp() + ":" + getSecondBackTapTimestamp());
                    listener.onActionRecognized(actionResult);
                }
            }
        }
        int count2 = checkDoubleTopTapTiming(timestamp);
        if (count2 == 2) {
            if (actionListener != null) {
                for (ActionListener listener : actionListener) {
                    ActionResult actionResult = new ActionResult("TopTap");
                    actionResult.setTimestamp(getFirstTopTapTimestamp() + ":" + getSecondTopTapTimestamp());
                    listener.onAction(actionResult);
                }
                horizontalFilter.updateCondition();
            }
        }
    }
}