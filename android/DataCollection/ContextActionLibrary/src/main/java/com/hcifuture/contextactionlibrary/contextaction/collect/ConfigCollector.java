package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.contextaction.context.ConfigContext;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

import androidx.annotation.RequiresApi;

public class ConfigCollector extends BaseCollector {
    private final TriggerConfig triggerConfig;
    private long last_nonimu = 0;

    @RequiresApi(api = Build.VERSION_CODES.N)
    public ConfigCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
        triggerConfig = new TriggerConfig()
                .setAudioLength(5000)
                .setBluetoothScanTime(10000)
                .setWifiScanTime(10000);
    }

    @Override
    public void onAction(ActionResult action) {
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
        long current_call = context.getTimestamp();
        String appendCommit = "Context: " + context.getContext() + "\n" +
                "Context timestamp: " + context.getTimestamp() + "\n" +
                "Context reason: " + context.getReason();

        if (ConfigContext.NEED_AUDIO.equals(context.getContext())) {
            String name = "Event_Audio";
            triggerAndUpload(CollectorManager.CollectorType.Audio, triggerConfig, name, appendCommit);
        } else if (ConfigContext.NEED_NONIMU.equals(context.getContext())) {
            if (current_call - last_nonimu >= 1000) {
                last_nonimu = current_call;
                String name = "Event_NonIMU";
                triggerAndUpload(CollectorManager.CollectorType.NonIMU, triggerConfig, name, appendCommit);
            }
        } else if (ConfigContext.NEED_SCAN.equals(context.getContext())) {
            String name = "Event_Scan";
            triggerAndUpload(CollectorManager.CollectorType.Bluetooth, triggerConfig, name, appendCommit);
            triggerAndUpload(CollectorManager.CollectorType.Wifi, triggerConfig, name, appendCommit);
        }
    }
}
