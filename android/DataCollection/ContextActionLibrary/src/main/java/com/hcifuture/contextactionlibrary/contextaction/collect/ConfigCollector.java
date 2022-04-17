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

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

import androidx.annotation.RequiresApi;

public class ConfigCollector extends BaseCollector {
    private TriggerConfig triggerConfig;

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
        if (ConfigContext.NEED_AUDIO.equals(context.getContext())) {
            type = CollectorManager.CollectorType.Audio;
        } else if (ConfigContext.NEED_NONIMU.equals(context.getContext())) {
            type = CollectorManager.CollectorType.NonIMU;
        }
        if (type != null) {
            triggerAndUpload(type, triggerConfig, "Config_" + type, "Sensor: " + type);
        }
    }
}
