package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.SensorEvent;

import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;

public class ExampleAction extends BaseAction {
    private LogCollector logCollector;
    private int counter;

    public ExampleAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, LogCollector logCollector) {
        super(context, config, requestListener, actionListener);
        this.logCollector = logCollector;
        this.counter = 0;
    }

    @Override
    public void start() {

    }

    @Override
    public void stop() {

    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
        if (counter == 0) {
            logCollector.addLog(data.getName() + " " + data.getTimestamp());
        }
        counter = (counter + 1) % 100;
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    /*
    @Override
    public void onIMUSensorChanged(SensorEvent event) {
        logCollector.addLog(event.sensor.getName() + " " + event.timestamp);
    }


    @Override
    public void onProximitySensorChanged(SensorEvent event) {

    }
     */

    @Override
    public void getAction() {

    }
}
