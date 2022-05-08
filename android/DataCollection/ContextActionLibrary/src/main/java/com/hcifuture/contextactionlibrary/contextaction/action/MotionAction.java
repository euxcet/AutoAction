package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class MotionAction extends BaseAction {

    private boolean inited = false;
    private float initCount = 0;

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
        if (!inited) {
            inited = true;
            initCount = data.getStepCounter();
        }
        int curCount = (int)(data.getStepCounter() - initCount);
        Log.e("Step counter", String.valueOf(curCount));
    }

    @Override
    public void getAction() {
        if (!isStarted) {
            return;
        }
    }
}
