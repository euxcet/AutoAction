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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import androidx.annotation.RequiresApi;

public class TimedCollector extends BaseCollector {

    @RequiresApi(api = Build.VERSION_CODES.N)
    public TimedCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger, List<Trigger.CollectorType> types) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
        for (Trigger.CollectorType type : types) {
            futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                    () -> clickTrigger.trigger(Collections.singletonList(type), new TriggerConfig().setAudioLength(5000)).whenComplete((msg, ex) -> {
                        File sensorFile = new File(clickTrigger.getRecentPath(type));
                        Log.e("TimedCollector", "Sensor type: " + type);
                        Log.e("TimedCollector", "uploadSensorData: " + sensorFile);
                        NetworkUtils.uploadCollectedData(mContext,
                                sensorFile,
                                0,
                                "Timed_" + type,
                                "TestUserId_cwh",
                                System.currentTimeMillis(),
                                "Sensor_commit",
                                new StringCallback() {
                                    @Override
                                    public void onSuccess(Response<String> response) {
                                        Log.e("TimedCollector", "Sensor collect & upload success");
                                    }
                                });
                    }),
                    0,
                    15000,
                    TimeUnit.MILLISECONDS));
        }
    }

    @Override
    public void onAction(ActionResult action) {
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
    }
}
