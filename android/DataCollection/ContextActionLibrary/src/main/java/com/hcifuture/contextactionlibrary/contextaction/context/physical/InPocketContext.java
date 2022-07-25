package com.hcifuture.contextactionlibrary.contextaction.context.physical;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.NonIMUCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class InPocketContext extends BaseContext {

    private String TAG = "InPocketContext";

    private final float[] accMark = new float[3];
    private final float[] magMark = new float[3];
    private final float[] rotationMatrix = new float[9];
    private final float[] orientationAngles = new float[3];
    private final int ORIENTATION_CHECK_NUMBER = 5;
    private float[][] orientationMark = new float[ORIENTATION_CHECK_NUMBER][3];
    private boolean oriOk = false;
    private boolean lightOk = false;

    private boolean keep3s = false;
    private int lastOri = 0;
    private long lastDownTimestamp = 0, lastUpTimestamp = Long.MAX_VALUE, lastZeroTimestamp = 0;
    private Queue<Long> downTimestamps = new LinkedList<>();

    private NonIMUCollector nonIMUCollector;

    public InPocketContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener, NonIMUCollector nonIMUCollector, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, contextListener, scheduledExecutorService, futureList);
        this.nonIMUCollector = nonIMUCollector;
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Context is already started.");
            return;
        }
        isStarted = true;
        oriOk = false;
        lightOk = false;
    }

    @Override
    public synchronized void stop() {
        if (!isStarted) {
            Log.d(TAG, "Context is already stopped");
            return;
        }
        isStarted = false;
    }

    private void updateOrientationAngles() {
        SensorManager.getRotationMatrix(rotationMatrix, null, accMark, magMark);
        SensorManager.getOrientation(rotationMatrix, orientationAngles);
        // orientationAngles: 方位角，俯仰角，倾侧角
        for (int i = 0; i < ORIENTATION_CHECK_NUMBER - 1; i++)
            System.arraycopy(orientationMark[i + 1], 0, orientationMark[i], 0, 3);
        System.arraycopy(orientationAngles, 0, orientationMark[ORIENTATION_CHECK_NUMBER - 1], 0, 3);
    }

    private int checkOrientation() {
        int downNum = 0, upNum = 0;
        for (int i = 0; i < ORIENTATION_CHECK_NUMBER; i++) {
            if (orientationMark[i][1] > 0.5)
                downNum++;
            else if (orientationMark[i][1] < 0)
                upNum++;
        }
        if (downNum >= ORIENTATION_CHECK_NUMBER)
            return -1;
        if (upNum >= ORIENTATION_CHECK_NUMBER)
            return 1;
        return 0;
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
        Heart.getInstance().newContextAliveEvent(getConfig().getContext(), data.getTimestamp());
        switch (data.getType()) {
            case Sensor.TYPE_ACCELEROMETER:
                for (int i = 0; i < 3; i++)
                    accMark[i] = data.getValues().get(i);
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                for (int i = 0; i < 3; i++)
                    magMark[i] = data.getValues().get(i);
                updateOrientationAngles();
                int curOri = checkOrientation();
                // check is down
                if (curOri == -1) {
                    if (lastOri != -1) {
                        if (data.getTimestamp() - lastDownTimestamp > 2 * 1e9)
                            downTimestamps.clear();
                        if (lastUpTimestamp > lastDownTimestamp)
                            downTimestamps.add(data.getTimestamp());
                        lastDownTimestamp = data.getTimestamp();
                        while (downTimestamps.size() > 0 && data.getTimestamp() - downTimestamps.peek() > 8 * 1e9)
                            downTimestamps.remove();
                        if (downTimestamps.size() >= 4)
                            keep3s = true;
                    }
                    if (!oriOk) {
                        oriOk = true;
                        if (contextListener != null) {
                            for (ContextListener listener : contextListener) {
                                listener.onContext(new ContextResult("InPocket"));
                            }
                        }
                        nonIMUCollector.openLightSensor();
                    }
                }
                // check is up
                else if (curOri == 1){
                    if (lastOri != 1) {
                        lastUpTimestamp = data.getTimestamp();
                    }
                    boolean changeStatus = false;
                    if (!keep3s) {
                        changeStatus = true;
                    }
                    if (keep3s && (data.getTimestamp() - lastDownTimestamp > 1.5 * 1e9 && data.getTimestamp() - lastZeroTimestamp > 1.5 * 1e9)) {
                        keep3s = false;
                        changeStatus = true;
                    }
                    if (!lightOk && changeStatus) {
                        if (oriOk) {
                            oriOk = false;
                            if (contextListener != null) {
                                for (ContextListener listener : contextListener) {
                                    listener.onContext(new ContextResult("OutOfPocket"));
                                }
                            }
                            nonIMUCollector.closeLightSensor();
                            nonIMUCollector.closeProxSensor();
                        }
                    }
                }
                else
                    lastZeroTimestamp = data.getTimestamp();
                lastOri = curOri;
                break;
            default:
                break;
        }
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {
        switch (data.getType()) {
            case Sensor.TYPE_LIGHT:
                if (data.getEnvironmentBrightness() < 1)
                    nonIMUCollector.openProxSensor();
                else {
                    lightOk = false;
                    nonIMUCollector.closeProxSensor();
                }
                break;
            case Sensor.TYPE_PROXIMITY:
                if (data.getProximity() < 4)
                    lightOk = true;
                else
                    lightOk = false;
                break;
            default:
                break;
        }
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {

    }

    @Override
    public void onBroadcastEvent(BroadcastEvent event) {

    }

    @Override
    public void onExternalEvent(Bundle bundle) {

    }

    @Override
    public void getContext() {
//        if (!isStarted) {
//            return;
//        }
//        if (contextListener != null) {
//            for (ContextListener listener: contextListener) {
//                if (oriOk)
//                    listener.onContext(new ContextResult("InPocket"));
//                else
//                    listener.onContext(new ContextResult("OutOfPocket"));
//            }
//        }
    }

    @Override
    public String getName() {
        return "InPocketContext";
    }
}