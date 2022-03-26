package com.hcifuture.contextactionlibrary.data;

import android.content.Context;
import android.hardware.SensorEvent;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.contextaction.action.BaseAction;
import com.hcifuture.contextactionlibrary.contextaction.context.BaseContext;
import com.hcifuture.shared.communicate.event.BroadcastEvent;

import java.util.List;

public class BroadcastEventManager extends BaseSensorManager {
    public BroadcastEventManager(Context context, String name, List<BaseAction> actions, List<BaseContext> contexts) {
        super(context, name, actions, contexts);
    }

    @Override
    public List<Integer> getSensorTypeList() {
        return null;
    }

    @Override
    public void start() {

    }

    @Override
    public void stop() {

    }

    @Override
    public void onSensorChangedDex(SensorEvent event) {

    }

    @Override
    public void onAccessibilityEventDex(AccessibilityEvent event) {

    }

    @Override
    public void onBroadcastEventDex(BroadcastEvent event) {
        for (BaseContext context: contexts) {
            context.onBroadcastEvent(event);
        }
    }
}
