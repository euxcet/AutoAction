package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.contextaction.action.PocketAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TapTapAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TopTapAction;
import com.hcifuture.contextactionlibrary.contextaction.action.UploadDataAction;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class UploadDataCollector extends BaseCollector {

    LogCollector logCollector;
    private long lastUploadTime = 0;

    public UploadDataCollector(Context context, ScheduledExecutorService scheduledExecutorService,
                               List<ScheduledFuture<?>> futureList, RequestListener requestListener,
                               ClickTrigger clickTrigger, Uploader uploader, LogCollector uploadDataLogCollector) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
        logCollector = uploadDataLogCollector;
    }

    private long uploadInterval = 24*3600*1000; //默认一天上传一次数据
    private String folderName = "UploadData";

    public void setInterval(long interval) {
        this.uploadInterval = interval;
    }

    public void setName(String name) {
        this.folderName = name;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        if (action.getAction().equals(UploadDataAction.UPLOAD_ACTION)) {
            long time = System.currentTimeMillis();
            Heart.getInstance().newCollectorAliveEvent(getName(), time);
            try {
                if (time - lastUploadTime > uploadInterval) {
                    triggerAndUpload(logCollector, new TriggerConfig(), folderName, "upload")
                            .thenAccept(v -> {
                                lastUploadTime = time;
                                logCollector.eraseLog(v.getLogLength());
                            });
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
    }

    @Override
    public String getName() {
        return "UploadDataCollector";
    }
}
