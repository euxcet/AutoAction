package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.os.Bundle;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class MotionAction extends BaseAction {

    public static String NEED_POSITION = "action.motion.need_position";

    private final int threshold = 500;

    private boolean inited = false;
    private int initCount = 0;
    private int lastCount = 0;

    public MotionAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, actionListener, scheduledExecutorService, futureList);
    }

    @Override
    public void start() {
        isStarted = true;
    }

    @Override
    public void stop() {
        isStarted = false;
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {
        if (data.getType() == Sensor.TYPE_STEP_COUNTER) {
            Heart.getInstance().newActionAliveEvent(getConfig().getAction(), data.getTimestamp());
            int curCount = (int)data.getStepCounter();
            if (!inited) {
                inited = true;
                initCount = curCount;
                lastCount = curCount;
            }
            // notify action to collect GPS & Location data
            if (curCount - lastCount >= threshold) {
                if (actionListener != null) {
                    Log.e("MotionAction", "notifyAction: " + NEED_POSITION);
                    long timestamp = System.currentTimeMillis();
                    for (ActionListener listener : actionListener) {
                        ActionResult actionResult = new ActionResult(NEED_POSITION, "steps reach threshold: " + threshold);
                        actionResult.setTimestamp(timestamp);
                        actionResult.getExtras().putInt("curCount", curCount);
                        actionResult.getExtras().putInt("lastCount", lastCount);
                        actionResult.getExtras().putInt("threshold", threshold);
                        actionResult.getExtras().putInt("initCount", initCount);
                        listener.onAction(actionResult);
                    }
                }
                lastCount = curCount;
            }
        }
    }

    @Override
    public void onExternalEvent(Bundle bundle) {

    }

    @Override
    public void getAction() {
        if (!isStarted) {
            return;
        }
    }

    @Override
    public String getName() {
        return "MotionAction";
    }
}
