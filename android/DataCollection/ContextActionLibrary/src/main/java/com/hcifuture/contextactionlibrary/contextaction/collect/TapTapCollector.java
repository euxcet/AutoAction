package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

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
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class TapTapCollector extends BaseCollector {
    public TapTapCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        if (!(action.getAction().equals("TapTap") || action.getAction().equals("TopTap") || action.getAction().equals("Pocket")))
            return;
        if (clickTrigger != null && scheduledExecutorService != null) {
            futureList.add(scheduledExecutorService.schedule(() -> {
                try {
                    clickTrigger.trigger(Collections.singletonList(Trigger.CollectorType.CompleteIMU), new TriggerConfig()).whenComplete((v, t) -> {
                        File imuFile = new File(clickTrigger.getRecentIMUPath());
                        Log.e("TapTapCollector", imuFile.getAbsolutePath());
                        NetworkUtils.uploadCollectedData(mContext,
                                imuFile,
                                0,
                                action.getAction(),
                                getMacMoreThanM(),
                                System.currentTimeMillis(),
                                action.getAction() + ":" + action.getReason() + ":" + action.getTimestamp(),
                                new StringCallback() {
                                    @Override
                                    public void onSuccess(Response<String> response) {
                                        Log.e("TapTapCollector", "Success");
                                    }
                                });

                    });
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }, 20000L, TimeUnit.MILLISECONDS));
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
        if (!context.getContext().equals("UserAction"))
            return;
        if (clickTrigger != null && scheduledExecutorService != null) {
            // save
            clickTrigger.triggerShortIMU(800, 200).whenComplete((msg, ex) -> {
                // upload when done
                File imuFile = new File(clickTrigger.getRecentIMUPath());
                Log.e("TapTapCollector", imuFile.getAbsolutePath());
                NetworkUtils.uploadCollectedData(mContext,
                        imuFile,
                        0,
                        context.getContext(),
                        getMacMoreThanM(),
                        System.currentTimeMillis(),
                        context.getContext() + ":" + context.getTimestamp(),
                        new StringCallback() {
                            @Override
                            public void onSuccess(Response<String> response) {
                                Log.e("TapTapCollector", "Success");
                            }
                        });
            });
        }
    }
}
