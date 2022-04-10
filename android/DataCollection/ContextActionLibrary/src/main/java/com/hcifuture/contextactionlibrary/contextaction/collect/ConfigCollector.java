package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.hcifuture.contextactionlibrary.collect.collector.LogCollector;
import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import androidx.annotation.RequiresApi;

public class ConfigCollector extends BaseCollector {
    @RequiresApi(api = Build.VERSION_CODES.N)
    public ConfigCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger, LogCollector logCollector) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> clickTrigger.trigger(logCollector).whenComplete((msg, ex) -> {
                    File logFile = new File(logCollector.getRecentPath());
                    Log.e("ConfigCollector", "uploadCollectedData: "+logFile.toString());
                    NetworkUtils.uploadCollectedData(mContext,
                            logFile,
                            0,
                            "Config",
                            getMacMoreThanM(),
                            System.currentTimeMillis(),
                            "ConfigLog_commit",
                            new StringCallback() {
                                @Override
                                public void onSuccess(Response<String> response) {
                                    Log.e("ConfigLogger", "Success");
                                }
                            });
                }),
                5000,
                60000,
                TimeUnit.MILLISECONDS));
    }

    @Override
    public void onAction(ActionResult action) {
    }

    @Override
    public void onContext(ContextResult context) {

    }
}
