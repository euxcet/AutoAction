package com.example.contextactionlibrary.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

import com.example.contextactionlibrary.contextaction.action.ActionBase;
import com.example.contextactionlibrary.contextaction.context.ContextBase;
import com.example.ncnnlibrary.communicate.event.ButtonActionEvent;

import java.util.List;

public class AccessibilityEventManager extends MySensorManager {
    public AccessibilityEventManager(Context context, String name, List<ActionBase> actions, List<ContextBase> contexts) {
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
        for (ContextBase context: contexts) {
            context.onAccessibilityEvent(event);
        }
    }

    @Override
    public void onButtonActionEventDex(ButtonActionEvent event) {

    }
}
