package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class InformationalContextCollector extends BaseCollector {

    @RequiresApi(api = Build.VERSION_CODES.N)
    public InformationalContextCollector(Context context, ScheduledExecutorService scheduledExecutorService,
                                         List<ScheduledFuture<?>> futureList, RequestListener requestListener,
                                         ClickTrigger clickTrigger, Uploader uploader,
                                         LogCollector logCollector) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
    }

    @Override
    public void onAction(ActionResult action) {
    }



    @Override
    public void onContext(ContextResult context) {

    }

    @Override
    public String getName() {
        return "InformationalContextCollector";
    }


}
