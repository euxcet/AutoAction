package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.hcifuture.contextactionlibrary.collect.collector.LogCollector;
import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;
import com.hcifuture.contextactionlibrary.collect.trigger.TriggerConfig;
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
import java.util.concurrent.TimeUnit;

import androidx.annotation.RequiresApi;

public class TimedCollector extends BaseCollector {

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector(Context context, ScheduledExecutorService scheduledExecutorService,
                          List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleTimedSensorUpload(Trigger.CollectorType type, TriggerConfig triggerConfig, long period, long delay) {
        if (type == Trigger.CollectorType.Log) {
            Log.e("TimedCollector", "Do not pass CollectorType.Log in newTimedSensorUpload(), it will be ignored.");
            return this;
        }
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> clickTrigger.trigger(Collections.singletonList(type), triggerConfig).whenComplete((msg, ex) -> {
                    try {
                        File sensorFile = new File(clickTrigger.getRecentPath(type));
                        Log.e("TimedCollector", "Sensor type: " + type);
                        Log.e("TimedCollector", "Sensor file: " + sensorFile);
                        NetworkUtils.uploadCollectedData(mContext,
                                sensorFile,
                                0,
                                "Timed_" + type,
                                getUserID(),
                                System.currentTimeMillis(),
                                "Sensor: " + type,
                                new StringCallback() {
                                    @Override
                                    public void onSuccess(Response<String> response) {
                                        Log.e("TimedCollector", type + " sensor upload success");
                                    }
                                });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }), delay, period, TimeUnit.MILLISECONDS)
        );
        return this;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleTimedLogUpload(LogCollector logCollector, long period, long delay, String name) {
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> clickTrigger.trigger(logCollector, new TriggerConfig()).whenComplete((msg, ex) -> {
                    try {
                        File file = new File(logCollector.getRecentPath());
                        Log.e("TimedCollector", "Log name: " + name);
                        Log.e("TimedCollector", "Log file: " + file);
                        NetworkUtils.uploadCollectedData(mContext,
                                file,
                                0,
                                "Timed_" + name,
                                getUserID(),
                                System.currentTimeMillis(),
                                "Log: " + name,
                                new StringCallback() {
                                    @Override
                                    public void onSuccess(Response<String> response) {
                                        logCollector.eraseLog();
                                        Log.e("TimedCollector", name + " log upload success && locally erased");
                                    }
                                });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }), delay, period, TimeUnit.MILLISECONDS)
        );
        return this;
    }

    @Override
    public void onAction(ActionResult action) {
    }

    @Override
    public void onContext(ContextResult context) {
    }
}
