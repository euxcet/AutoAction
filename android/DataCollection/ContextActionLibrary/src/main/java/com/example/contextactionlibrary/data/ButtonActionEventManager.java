package com.example.contextactionlibrary.data;

import android.content.Context;
import android.hardware.SensorEvent;
import android.view.accessibility.AccessibilityEvent;

import com.example.contextactionlibrary.contextaction.action.ActionBase;
import com.example.contextactionlibrary.contextaction.context.ContextBase;
import com.example.ncnnlibrary.communicate.event.ButtonActionEvent;

import java.util.List;

public class ButtonActionEventManager extends MySensorManager {
    public ButtonActionEventManager(Context context, String name, List<ActionBase> actions, List<ContextBase> contexts) {
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
    public void onButtonActionEventDex(ButtonActionEvent event) {
        for (ContextBase context: contexts) {
            context.onButtonActionEvent(event);
        }
    }
}
