package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.contextaction.action.TapTapAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TopTapAction;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.async.IMUCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class TapTapCollector extends BaseCollector {
    private String markTimestamp;
    private CompletableFuture<List<String>> lastFuture;

    public TapTapCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        if (action.getAction().equals(TapTapAction.ACTION) || action.getAction().equals(TopTapAction.ACTION)) {
            lastFuture = clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.IMU), new TriggerConfig());
        } else if (action.getAction().equals(TapTapAction.ACTION_UPLOAD) || action.getAction().equals(TopTapAction.ACTION_UPLOAD)) {
            if (lastFuture != null) {
                // TODO: upload with timestamp markTimestamp
            }
        } else if (action.getAction().equals(TapTapAction.ACTION_RECOGNIZED) || action.getAction().equals(TopTapAction.ACTION_RECOGNIZED)) {
            markTimestamp = action.getTimestamp();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
        if (!context.getContext().equals("UserAction")) {
            return;
        }
        if (clickTrigger != null) {
            clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.IMU),
                    new TriggerConfig().setImuHead(800).setImuTail(200));
        }
        /*
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
         */
    }
}
