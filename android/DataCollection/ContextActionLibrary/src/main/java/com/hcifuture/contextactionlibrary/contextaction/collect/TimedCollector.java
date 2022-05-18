package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import androidx.annotation.RequiresApi;

public class TimedCollector extends BaseCollector {

    public static String TIMED_FIXED_RATE = "timed_collector.fixed_rate";
    public static String TIMED_FIXED_RATE_LOG = "timed_collector.fixed_rate.log";
    public static String TIMED_FIXED_DELAY = "timed_collector.fixed_delay";

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector(Context context, ScheduledExecutorService scheduledExecutorService,
                          List<ScheduledFuture<?>> futureList, RequestListener requestListener,
                          ClickTrigger clickTrigger, Uploader uploader) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleFixedRateUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, long period, long initialDelay, String name) {
        if (type == CollectorManager.CollectorType.Log) {
            Log.e("TimedCollector", "Do not pass CollectorType.Log in scheduleFixedRateUpload(), it will be ignored.");
            return this;
        }
        AtomicInteger idx = new AtomicInteger(0);
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> {
                    try {
                        ContextResult contextResult = new ContextResult(TIMED_FIXED_RATE, "Fixed rate upload");
                        contextResult.setTimestamp(System.currentTimeMillis());
                        contextResult.getExtras().putLong("period", period);
                        contextResult.getExtras().putLong("initialDelay", initialDelay);
                        contextResult.getExtras().putInt("idx", idx.getAndIncrement());
                        triggerAndUpload(type, triggerConfig, name, "", contextResult);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                },
                initialDelay, period, TimeUnit.MILLISECONDS)
        );
        Log.e("TimedCollector", "register fixed rate upload: " + type.name() +
                " period: " + period +
                " initialDelay: " + initialDelay +
                " name: " + name);
        return this;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleFixedDelayUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, long delay, long initialDelay, String name) {
        if (type == CollectorManager.CollectorType.Log) {
            Log.e("TimedCollector", "Do not pass CollectorType.Log in scheduleFixedDelayUpload(), it will be ignored.");
            return this;
        }
        AtomicInteger idx = new AtomicInteger(0);
        futureList.add(scheduledExecutorService.scheduleWithFixedDelay(
                () -> {
                    try {
                        ContextResult contextResult = new ContextResult(TIMED_FIXED_DELAY, "Fixed delay upload");
                        contextResult.setTimestamp(System.currentTimeMillis());
                        contextResult.getExtras().putLong("delay", delay);
                        contextResult.getExtras().putLong("initialDelay", initialDelay);
                        contextResult.getExtras().putInt("idx", idx.getAndIncrement());
                        triggerAndUpload(type, triggerConfig, name, "", contextResult).join();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                },
                initialDelay, delay, TimeUnit.MILLISECONDS)
        );
        Log.e("TimedCollector", "register fixed delay upload: " + type.name() +
                " delay: " + delay +
                " initialDelay: " + initialDelay +
                " name: " + name);
        return this;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleTimedLogUpload(LogCollector logCollector, long period, long initialDelay, String name) {
        AtomicInteger idx = new AtomicInteger(0);
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> {
                    try {
                        ContextResult contextResult = new ContextResult(TIMED_FIXED_RATE_LOG, "Fixed rate log upload");
                        contextResult.setTimestamp(System.currentTimeMillis());
                        contextResult.getExtras().putLong("period", period);
                        contextResult.getExtras().putLong("initialDelay", initialDelay);
                        contextResult.getExtras().putInt("idx", idx.getAndIncrement());
                        triggerAndUpload(logCollector, new TriggerConfig(), name, "", contextResult)
                                .thenAccept(v -> logCollector.eraseLog(v.getLogLength()));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                },
                initialDelay, period, TimeUnit.MILLISECONDS)
        );
        Log.e("TimedCollector", "register fixed rate upload Log" +
                " period: " + period +
                " initialDelay: " + initialDelay +
                " name: " + name);
        return this;
    }

    @Override
    public void onAction(ActionResult action) {
    }

    @Override
    public void onContext(ContextResult context) {
    }
}
