package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
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

public class ExampleCollector extends BaseCollector {
    public ExampleCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger, LogCollector logCollector) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
        futureList.add(scheduledExecutorService.scheduleAtFixedRate(
                () -> {
                    clickTrigger.trigger(logCollector);
                    futureList.add(scheduledExecutorService.schedule(() -> {
                        File logFile = new File(logCollector.getRecentPath());
                        NetworkUtils.uploadCollectedData(mContext,
                                logFile,
                                0,
                                "Log0",
                                getMacMoreThanM(),
                                System.currentTimeMillis(),
                                "Log0_commit",
                                new StringCallback() {
                                    @Override
                                    public void onSuccess(Response<String> response) {
                                        Log.e("TapTapCollector", "Success");
                                    }
                                });
                    }, 0, TimeUnit.MILLISECONDS));
                },
                0,
                5000,
                TimeUnit.MILLISECONDS));
    }

    @Override
    public void onAction(ActionResult action) {
    }

    @Override
    public void onContext(ContextResult context) {

    }
}
