package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.os.Bundle;

import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class UploadDataAction extends BaseAction {

    public UploadDataAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
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

    }

    @Override
    public void onExternalEvent(Bundle bundle) {
        if (bundle.getString("type", "").equals("UploadData")) {
            String bundleName = bundle.getString("name");
            String upLoadName = config.getString("name");
            if (bundleName != null && bundleName.equals(upLoadName)) {
                String data = bundle.getString("data", "");
                getLogCollector().addLog(data);
            }
        }
    }

    @Override
    public void getAction() {

    }

    @Override
    public String getName() {
        return null;
    }
}
