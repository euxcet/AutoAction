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
    public TimedCollector scheduleTimedSensorUpload(Trigger.CollectorType type, TriggerConfig triggerConfig, long period, long initialDelay) {
        if (type == Trigger.CollectorType.Log) {
            Log.e("TimedCollector", "Do not pass CollectorType.Log in scheduleTimedSensorUpload(), it will be ignored.");
            return this;
        }
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> triggerAndUpload(type, triggerConfig, "Timed_" + type, "Sensor: " + type),
                initialDelay, period, TimeUnit.MILLISECONDS)
        );
        return this;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector scheduleTimedLogUpload(LogCollector logCollector, long period, long initialDelay, String name) {
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> triggerAndUpload(logCollector, new TriggerConfig(), "Timed_"+name, "Log: "+name)
                        .whenComplete((msg, ex) -> logCollector.eraseLog()),
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
