package com.hcifuture.contextactionlibrary.sensor.distributor;

import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.contextaction.action.BaseAction;
import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorListener;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.SensorType;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;

public class DataDistributor implements CollectorListener {
    private List<BaseAction> actions;
    private List<BaseContext> contexts;
    private AtomicBoolean isRunning;
    private Lock contextLock;
    public DataDistributor(List<BaseAction> actions, List<BaseContext> contexts, Lock contextLock) {
        this.actions = actions;
        this.contexts = contexts;
        this.isRunning = new AtomicBoolean(true);
        this.contextLock = contextLock;
    }

    public void start() {
        isRunning.set(true);
    }

    public void stop() {
        isRunning.set(false);
    }

    public void onExternalEvent(Bundle bundle) {
//        if (isRunning.get()) {
            contextLock.lock();
            try {
                for (BaseAction action : actions) {
                    action.onExternalEvent(bundle);
                }
                for (BaseContext context : contexts) {
                    context.onExternalEvent(bundle);
                }
            } finally {
                contextLock.unlock();
            }
//        }
    }

    public void onBroadcastEvent(BroadcastEvent event) {
//        if (isRunning.get()) {
            contextLock.lock();
            try {
                for (BaseContext context : contexts) {
                    context.onBroadcastEvent(event);
                }
            } finally {
                contextLock.unlock();
            }
//        }
    }

    public void onAccessibilityEvent(AccessibilityEvent event) {
//        if (isRunning.get()) {
            contextLock.lock();
            try {
                for (BaseContext context : contexts) {
                    context.onAccessibilityEvent(event);
                }
            } finally {
                contextLock.unlock();
            }
//        }
    }

    @Override
    public void onSensorEvent(Data data) {
        if (isRunning.get()) {
            contextLock.lock();
            try {
                for (BaseAction action : actions) {
                    List<SensorType> types = action.getConfig().getSensorType();
                    switch (data.dataType()) {
                        case SingleIMUData:
                            if (types.contains(SensorType.IMU)) {
                                action.onIMUSensorEvent((SingleIMUData) data);
                            }
                            break;
                        case NonIMUData:
                            if (types.contains(SensorType.PROXIMITY)) {
                                action.onNonIMUSensorEvent((NonIMUData) data);
                            }
                            break;
                        default:
                            break;
                    }
                }
                for (BaseContext context : contexts) {
                    List<SensorType> types = context.getConfig().getSensorType();
                    switch (data.dataType()) {
                        case SingleIMUData:
                            if (types.contains(SensorType.IMU)) {
                                context.onIMUSensorEvent((SingleIMUData) data);
                            }
                            break;
                        case NonIMUData:
                            if (types.contains(SensorType.PROXIMITY)) {
                                context.onNonIMUSensorEvent((NonIMUData) data);
                            }
                            break;
                        default:
                            break;
                    }
                }
            } finally {
                contextLock.unlock();
            }
        }
    }
}
