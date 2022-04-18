package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.util.Log;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.hcifuture.contextactionlibrary.contextaction.action.tapfilter.HorizontalFilter;
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

public class TopTapAction extends BaseAction {

    private String TAG = "TopTapAction";

    public static String ACTION = "action.toptap.action";
    public static String ACTION_UPLOAD = "action.toptap.action.upload";
    public static String ACTION_RECOGNIZED = "action.toptap.action.recognized";

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
    private long lastTimestamp = 0L;

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
    private Deque<Long> backTapTimestamps = new ArrayDeque();
    private Deque<Long> topTapTimestamps = new ArrayDeque();
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
        // just for horizontal / static cases' record && upload
        horizontalFilter.onSensorChanged(data);
        if (horizontalFilter.passWithDelay(data.getTimestamp()) == -1) {
            ActionResult actionResult = new ActionResult(ACTION_UPLOAD);
            actionResult.setReason("Static");
            for (ActionListener listener : actionListener) {
                listener.onAction(actionResult);
            }
        }

        if (data.getType() != Sensor.TYPE_GYROSCOPE && data.getType() != Sensor.TYPE_LINEAR_ACCELERATION)
            return;
        result = 0;
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
            lowpassKeyNegative.init(0.0F);
            highpassKeyNegative.init(0.0F);
        } else {
            if (data.getType() == Sensor.TYPE_LINEAR_ACCELERATION)
                processAccAndKeySignal(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), data.getTimestamp(), SAMPLINGINTERVALNS);
            else
                processGyro(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), SAMPLINGINTERVALNS);
            recognizeTapML();
            if (result == 1) {
                backTapTimestamps.addLast(data.getTimestamp());
            }
            else if (result == 2) {
                topTapTimestamps.addLast(data.getTimestamp());
            }
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
        int size = (int)(WINDOW_NS / samplingInterval);

        while(xsAcc.size() > size) {
            xsAcc.remove(0);
            ysAcc.remove(0);
            zsAcc.remove(0);
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
        synchronized (list) {
            for (int i = 0; i < seqLength; i++) {
                if (i + startIdx >= list.size())
                    res.add(0.0F);
                else
                    res.add(scale * list.get(i + startIdx));
            }
        }
    }

    private int checkDoubleBackTapTiming(long timestamp) {
        Iterator iter = backTapTimestamps.iterator();
        while (iter.hasNext()) {
            if (timestamp - (Long) iter.next() > 500000000L) {
                iter.remove();
            }
        }
        int res = 0;
        if (!backTapTimestamps.isEmpty()) {
            iter = backTapTimestamps.iterator();
            while (true) {
                if (!iter.hasNext()) {
                    res = 1;
                    break;
                }
                doubleBackTapTimestamps[1] = (Long)backTapTimestamps.getLast();
                doubleBackTapTimestamps[0] = (Long)iter.next();
                if (doubleBackTapTimestamps[1] - doubleBackTapTimestamps[0] > 100000000L) {
                    backTapTimestamps.clear();
                    res = 2;
                    break;
                }
            }
        }
        return res;
    }

    private int checkDoubleTopTapTiming(long timestamp) {
        Iterator iter = topTapTimestamps.iterator();
        while (iter.hasNext()) {
            if (timestamp - (Long) iter.next() > 500000000L) {
                iter.remove();
            }
        }
        int res = 0;
        if (!topTapTimestamps.isEmpty()) {
            iter = topTapTimestamps.iterator();
            while (true) {
                if (!iter.hasNext()) {
                    res = 1;
                    break;
                }
                doubleTopTapTimestamps[1] = (Long)topTapTimestamps.getLast();
                doubleTopTapTimestamps[0] = (Long)iter.next();
                if (doubleTopTapTimestamps[1] - doubleTopTapTimestamps[0] > 100000000L) {
                    topTapTimestamps.clear();
                    res = 2;
                    break;
                }
            }
        }
        return res;
    }

    public long getFirstBackTapTimestamp() {
        return doubleBackTapTimestamps[0];
    }

    public long getSecondBackTapTimestamp() {
        return doubleBackTapTimestamps[1];
    }

    public long getFirstTopTapTimestamp() {
        return doubleTopTapTimestamps[0];
    }

    public long getSecondTopTapTimestamp() {
        return doubleTopTapTimestamps[1];
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
        int count1 = checkDoubleBackTapTiming(lastTimestamp);
        if (count1 == 2) {
            if (actionListener != null) {
                for (ActionListener listener: actionListener) {
                    ActionResult actionResult = new ActionResult(ACTION_RECOGNIZED);
                    actionResult.setTimestamp(getFirstBackTapTimestamp() + ":" + getSecondBackTapTimestamp());
                    listener.onAction(actionResult);
                    TapTapAction.onConfirmed();
                }
            }
        }
        int count2 = checkDoubleTopTapTiming(lastTimestamp);
        if (count2 == 2) {
            if (actionListener != null) {
                for (ActionListener listener : actionListener) {
                    ActionResult actionResult = new ActionResult(ACTION_RECOGNIZED);
                    actionResult.setTimestamp(getFirstTopTapTimestamp() + ":" + getSecondTopTapTimestamp());
                    listener.onAction(actionResult);
                    actionResult.setAction(ACTION);
                    listener.onAction(actionResult);
                }
                horizontalFilter.updateCondition();
            }
        }
    }
}
