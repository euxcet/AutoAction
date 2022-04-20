package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.Trigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

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
    public TimedCollector scheduleFixedRateUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, long period, long initialDelay) {
        if (type == CollectorManager.CollectorType.Log) {
            Log.e("TimedCollector", "Do not pass CollectorType.Log in scheduleFixedRateUpload(), it will be ignored.");
            return this;
        }
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> {
                    try {
                        triggerAndUpload(type, triggerConfig, "Timed_" + type, "Fixed rate upload: " + period);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                },
                initialDelay, period, TimeUnit.MILLISECONDS)
        );
        return this;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleFixedDelayUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, long delay, long initialDelay) {
        if (type == CollectorManager.CollectorType.Log) {
            Log.e("TimedCollector", "Do not pass CollectorType.Log in scheduleFixedDelayUpload(), it will be ignored.");
            return this;
        }
        futureList.add(scheduledExecutorService.scheduleWithFixedDelay(
                () -> {
                    try {
                        triggerAndUpload(type, triggerConfig, "Timed_" + type, "Fixed delay upload: " + delay).join();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                },
                initialDelay, delay, TimeUnit.MILLISECONDS)
        );
        return this;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleTimedLogUpload(LogCollector logCollector, long period, long initialDelay, String name) {
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> {
                    try {
                        triggerAndUpload(logCollector, new TriggerConfig(), "Timed_" + name, "Fixed rate upload: " + period + "\r\n" + "Log name: " + name)
                                .thenAccept(v -> logCollector.eraseLog(v.getLogLength()));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                },
                initialDelay, period, TimeUnit.MILLISECONDS));
        return this;
    }

    @Override
    public void onAction(ActionResult action) {
    }

    @Override
    public void onContext(ContextResult context) {
    }
}
