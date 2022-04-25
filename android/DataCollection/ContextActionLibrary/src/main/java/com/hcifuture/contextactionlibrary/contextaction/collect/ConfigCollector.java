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
    public ConfigCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger, LogCollector logCollector) {
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
        CollectorManager.CollectorType type = null;
        long current_call = context.getTimestamp();

        if (ConfigContext.NEED_AUDIO.equals(context.getContext())) {
            type = CollectorManager.CollectorType.Audio;
        } else if (ConfigContext.NEED_NONIMU.equals(context.getContext())) {
            if (current_call - last_nonimu >= 1000) {
                type = CollectorManager.CollectorType.NonIMU;
                last_nonimu = current_call;
            }
        } else if (ConfigContext.NEED_SCAN.equals(context.getContext())) {
            triggerAndUpload(Arrays.asList(CollectorManager.CollectorType.Bluetooth, CollectorManager.CollectorType.Wifi),
                    triggerConfig, "Event_Scan", "Context: " + context.getContext() + "\n" + "Context timestamp: " + context.getTimestamp());
        }

        if (type != null) {
            triggerAndUpload(type, triggerConfig, "Config_" + type, "Context: " + context.getContext() + "\n" + "Context timestamp: " + context.getTimestamp());
        }
    }
}
