package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class ActivityAction extends BaseAction {

    private String TAG = "ActivityAction";

    public static String ACTION_STATIC = "action.static.action";
    public static String ACTION_WALKING = "action.walking.action";
    public static String ACTION_RUNNING = "action.running.action";
    public static String ACTION_CYCLING = "action.cycling.action";
    public static String ACTION_OTHERS = "action.others.action";

    private boolean initialized = false;

    private static int seqLength = 500;
    private float[] imuInput = new float[seqLength * 6];
    private long[] imuSize = new long[]{1, 6, seqLength};
    private long[] lastTime = new long[2];
    private int skipNum = seqLength;
    private Module imuModule = null;
    private String prevActivity = ACTION_OTHERS;

    public ActivityAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, actionListener, scheduledExecutorService, futureList);
        init();
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    private void init() {
        if (initialized)
            return;

        try {
            imuModule = LiteModuleLoader.load(ContextActionContainer.getSavePath() + "activity.ptl");
            Log.i(TAG, "Model load succeed");
        } catch (Exception e) {
            Log.e(TAG, "Model load fail: " + e);
            e.printStackTrace();
        }

        prevActivity = ACTION_OTHERS;
        initialized = true;
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
    public void onIMUSensorEvent(SingleIMUData data) {
        if (data.getType() != Sensor.TYPE_GYROSCOPE && data.getType() != Sensor.TYPE_LINEAR_ACCELERATION)
            return;
        Heart.getInstance().newActionAliveEvent(getConfig().getAction(), data.getTimestamp());
        int idx = (data.getType() == Sensor.TYPE_LINEAR_ACCELERATION) ? 0 : 1;
        if (data.getTimestamp() < lastTime[idx] + 3 * 1e6)
            return;
        lastTime[idx] = data.getTimestamp();
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < seqLength - 1; j++)
                imuInput[i * seqLength + j] = imuInput[i * seqLength + j + 1];
        imuInput[(3 * idx + 1) * seqLength - 1] = data.getValues().get(0);
        imuInput[(3 * idx + 2) * seqLength - 1] = data.getValues().get(1);
        imuInput[(3 * idx + 3) * seqLength - 1] = data.getValues().get(2);
        if (idx == 0) {
            if (skipNum > 0) {
                skipNum--;
            } else {
                skipNum = seqLength;
                recognize();
            }
        }
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    @Override
    public void onExternalEvent(Bundle bundle) {

    }

    private void recognize() {
        Tensor imuInputTensor = Tensor.fromBlob(imuInput, imuSize);
        Tensor imuOutputTensor = imuModule.forward(IValue.from(imuInputTensor)).toTensor();
        float[] imuScores = imuOutputTensor.getDataAsFloatArray();
        int activity = 0;
        for (int i = 1; i < imuScores.length; i++) {
            if (imuScores[i] > imuScores[activity])
                activity = i;
        }
        Log.i(TAG, "Label: " + activity);
        String curActivity = ACTION_OTHERS;
        if (activity <= 3 || activity == 15 || activity == 16)
            curActivity = ACTION_WALKING;
        else if (activity == 4 || activity == 17)
            curActivity = ACTION_RUNNING;
        else if (activity == 5)
            curActivity = ACTION_CYCLING;
        else if (activity == 11 || activity == 12 || activity == 14)
            curActivity = ACTION_STATIC;
        if (prevActivity != curActivity) {
            prevActivity = curActivity;
            if (actionListener != null) {
                for (ActionListener listener : actionListener) {
                    ActionResult actionResult = new ActionResult(curActivity);
                    listener.onAction(actionResult);
                }
            }
        }
        Log.i(TAG, curActivity);
    }

    @Override
    public synchronized void getAction() {
        if (!isStarted)
            return;
    }
}
