package com.example.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.SensorEvent;
import android.util.Log;

import com.example.contextactionlibrary.data.Preprocess;
import com.google.android.systemui.columbus.sensors.TfClassifier;
import com.google.android.systemui.columbus.sensors.Util;

import java.util.ArrayList;
import java.util.List;

public class TapTapAction extends ActionBase {

    private String TAG = "TapTapAction";

    private List<Long> tapTimestamps = new ArrayList();
    private TfClassifier tflite;

    private Preprocess preprocess;

    public TapTapAction(Context context, ActionListener actionListener, int seqLength, String[] actions) {
        super(context, actionListener, seqLength, actions);
        tflite = new TfClassifier(mContext.getAssets(), "tap7cls_pixel4.tflite");
        preprocess = Preprocess.getInstance();
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Action is already started.");
            return;
        }
        isStarted = true;
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
    public void onAlwaysOnSensorChanged(SensorEvent event) {

    }

    private ArrayList<Float> getInput(int accIdx, int gyroIdx) {
        ArrayList<Float> res = new ArrayList<>();
        addFeatureData(preprocess.getXsAcc(), accIdx, 1, res);
        addFeatureData(preprocess.getYsAcc(), accIdx, 1, res);
        addFeatureData(preprocess.getZsAcc(), accIdx, 1, res);
        addFeatureData(preprocess.getXsGyro(), gyroIdx, 10, res);
        addFeatureData(preprocess.getYsGyro(), gyroIdx, 10, res);
        addFeatureData(preprocess.getZsGyro(), gyroIdx, 10, res);
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
    
    private int checkDoubleTapTiming(long timestamp) {
        // remove old timestamps
        int idx = 0;
        for (; idx < tapTimestamps.size(); idx++) {
            if (timestamp - tapTimestamps.get(idx) <= 500000000L)
                break;
        }
        tapTimestamps = tapTimestamps.subList(idx, tapTimestamps.size());

        // just check no tap && double tap now
        if (tapTimestamps.isEmpty())
            return 0;
        else {
            if (tapTimestamps.get(tapTimestamps.size() - 1) - tapTimestamps.get(0) > 100000000L) {
                tapTimestamps.clear();
                return 1;
            }
            return 0;
        }
    }

    @Override
    public void getAction() {
        if (!isStarted)
            return;
        int[] idxes = preprocess.shouldRunTapModel(seqLength);
        if (idxes[0] == -1)
            return;
        ArrayList<Float> input = getInput(idxes[0], idxes[1]);
        int result = Util.getMaxId((ArrayList)tflite.predict(input, 7).get(0));
        long timestamp = preprocess.getTimestamps().get(seqLength);
        if (result == 1) {
            tapTimestamps.add(timestamp);
            if (actionListener != null)
                actionListener.onAction(this, actions[checkDoubleTapTiming(timestamp)]);
        }
    }
}
