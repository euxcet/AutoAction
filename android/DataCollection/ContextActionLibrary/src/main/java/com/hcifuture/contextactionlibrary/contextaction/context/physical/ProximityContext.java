package com.hcifuture.contextactionlibrary.contextaction.context.physical;

import android.content.Context;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class ProximityContext extends BaseContext {
    private String TAG = "ProximityContext";

    private long lastTimestamp = -1;

    private static int proxThreshold = 4;
    private long lastNearTime = 0L;
    private long lastFarTime = 0L;
    private long lastRecognized = 0L;

    public ProximityContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, contextListener, scheduledExecutorService, futureList);
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Context is already started.");
            return;
        }
        isStarted = true;
    }

    @Override
    public synchronized void stop() {
        if (!isStarted) {
            Log.d(TAG, "Context is already stopped");
            return;
        }
        isStarted = false;
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {

    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {
        Heart.getInstance().newContextAliveEvent(getConfig().getContext(), data.getTimestamp());
        float prox = data.getProximity();
        if ((int)prox < proxThreshold) {
            lastNearTime = data.getProximityTimestamp();
        }
        else {
            lastFarTime = data.getProximityTimestamp();
        }
        lastTimestamp = data.getProximityTimestamp();
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {

    }

    @Override
    public void onBroadcastEvent(BroadcastEvent event) {

    }

    @Override
    public void getContext() {
        if (!isStarted) {
            return;
        }
        if (lastTimestamp - lastNearTime < 1.5 * 1e9 && lastTimestamp - lastRecognized > 5 * 1e9) {
            lastRecognized = lastTimestamp;
//            if (contextListener != null) {
//                for (ContextListener listener: contextListener) {
//                    listener.onContext(new ContextResult("Proximity"));
//                }
//            }
        }
    }
}
