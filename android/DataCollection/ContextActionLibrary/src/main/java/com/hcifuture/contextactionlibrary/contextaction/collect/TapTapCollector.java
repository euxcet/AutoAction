package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.util.Log;

import com.hcifuture.contextactionlibrary.collect.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
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

    @Override
    public void onAction(ActionResult action) {
        if (clickTrigger != null && scheduledExecutorService != null) {
            futureList.add(scheduledExecutorService.schedule(() -> {
                File imuFile = new File(clickTrigger.getRecentIMUPath());
                Log.e("TapTapCollector", imuFile.getAbsolutePath());
                NetworkUtils.uploadCollectedData(mContext,
                        imuFile,
                        0,
                        "TapTap",
                        getMacMoreThanM(),
                        System.currentTimeMillis(),
                        "Commit",
                        new StringCallback() {
                            @Override
                            public void onSuccess(Response<String> response) {
                                Log.e("TapTapCollector", "Success");
                            }
                        });
            }, 20000L, TimeUnit.MILLISECONDS));
        }
    }

    @Override
    public void onContext(ContextResult context) {

    }
}
