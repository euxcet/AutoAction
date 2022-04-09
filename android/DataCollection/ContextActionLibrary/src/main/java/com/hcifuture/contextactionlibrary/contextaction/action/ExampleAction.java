package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.SensorEvent;

import com.hcifuture.contextactionlibrary.collect.collector.LogCollector;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;

public class ExampleAction extends BaseAction {
    private LogCollector logCollector;

    public ExampleAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, LogCollector logCollector) {
        super(context, config, requestListener, actionListener);
        this.logCollector = logCollector;
    }

    @Override
    public void start() {

    }

    @Override
    public void stop() {

    }

    @Override
    public void onIMUSensorChanged(SensorEvent event) {
        logCollector.addLog(event.sensor.getName() + " " + event.timestamp);
    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {

    }

    @Override
    public void getAction() {

    }
}
