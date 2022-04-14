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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import androidx.annotation.RequiresApi;

public class ConfigCollector extends BaseCollector {
    List<Trigger.CollectorType> collect_types;
    TriggerConfig triggerConfig;

    @RequiresApi(api = Build.VERSION_CODES.N)
    public ConfigCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger, LogCollector logCollector) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> clickTrigger.trigger(logCollector, new TriggerConfig()).whenComplete((msg, ex) -> {
                    File logFile = new File(logCollector.getRecentPath());
                    Log.e("ConfigCollector", "uploadCollectedData: "+logFile);
                    NetworkUtils.uploadCollectedData(mContext,
                            logFile,
                            0,
                            "Config",
//                            getMacMoreThanM(),
                            "TestUserId_cwh",
                            System.currentTimeMillis(),
                            "ConfigLog_commit",
                            new StringCallback() {
                                @Override
                                public void onSuccess(Response<String> response) {
                                    Log.e("ConfigLogger", "Success");
                                    logCollector.eraseLog();
                                }
                            });
                }),
                5000,
                60000,
                TimeUnit.MILLISECONDS));

        collect_types = new ArrayList<>();
        collect_types.add(Trigger.CollectorType.Audio);
//        collect_types.add(Trigger.CollectorType.Bluetooth);
//        collect_types.add(Trigger.CollectorType.Wifi);
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
        if (ConfigContext.NEED_AUDIO.equals(context.getContext())) {
            for (Trigger.CollectorType type : collect_types) {
                clickTrigger.trigger(Collections.singletonList(type), triggerConfig).whenComplete((msg, ex) -> {
                    File sensorFile = new File(clickTrigger.getRecentPath(type));
                    Log.e("ConfigCollector", "Sensor type: " + type);
                    Log.e("ConfigCollector", "uploadSensorData: " + sensorFile);
                    NetworkUtils.uploadCollectedData(mContext,
                            sensorFile,
                            0,
                            "Config_"+type,
//                            getMacMoreThanM(),
                            "TestUserId_cwh",
                            System.currentTimeMillis(),
                            "Sensor_commit",
                            new StringCallback() {
                                @Override
                                public void onSuccess(Response<String> response) {
                                    Log.e("ConfigLogger", "Sensor collect & upload success");
                                }
                            });
                });
            }
        }
    }
}
