package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.hcifuture.contextactionlibrary.collect.collector.LogCollector;
import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;
import com.hcifuture.contextactionlibrary.collect.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.contextaction.context.ConfigContext;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

import androidx.annotation.RequiresApi;

public class ConfigCollector extends BaseCollector {
    TriggerConfig triggerConfig;

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
        Trigger.CollectorType type = null;
        if (ConfigContext.NEED_AUDIO.equals(context.getContext())) {
            type = Trigger.CollectorType.Audio;
        } else if (ConfigContext.NEED_NONIMU.equals(context.getContext())) {
            type = Trigger.CollectorType.NonIMU;
        }
        if (type != null) {
            triggerAndUpload(type, triggerConfig, "Config_"+type, "Sensor: "+type);
        }
    }
}
