package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.sensor.uploader.TaskMetaBean;
import com.hcifuture.contextactionlibrary.sensor.uploader.UploadTask;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.contextactionlibrary.utils.FileUtils;
import com.hcifuture.contextactionlibrary.utils.JSONUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.hcifuture.shared.communicate.result.Result;

import java.io.File;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class BaseCollector {
    protected Context mContext;
    protected RequestListener requestListener;
    protected ClickTrigger clickTrigger;
    protected ScheduledExecutorService scheduledExecutorService;
    protected List<ScheduledFuture<?>> futureList;
    private Uploader uploader;

    public BaseCollector(Context context, ScheduledExecutorService scheduledExecutorService,
                         List<ScheduledFuture<?>> futureList, RequestListener requestListener,
                         ClickTrigger clickTrigger, Uploader uploader) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.requestListener = requestListener;
        this.clickTrigger = clickTrigger;
        this.futureList = futureList;
        this.uploader = uploader;
    }

    public abstract void onAction(ActionResult action);

    public abstract void onContext(ContextResult context);

    protected static String getMacMoreThanM() {
        try {
            Enumeration enumeration = NetworkInterface.getNetworkInterfaces();
            while (enumeration.hasMoreElements()) {
                NetworkInterface networkInterface = (NetworkInterface) enumeration.nextElement();

                byte[] arrayOfByte = networkInterface.getHardwareAddress();
                if (arrayOfByte == null || arrayOfByte.length == 0) {
                    continue;
                }

                StringBuilder stringBuilder = new StringBuilder();
                for (byte b : arrayOfByte) {
                    stringBuilder.append(String.format("%02X:", b));
                }
                if (stringBuilder.length() > 0) {
                    stringBuilder.deleteCharAt(stringBuilder.length() - 1);
                }
                String str = stringBuilder.toString();
                if (networkInterface.getName().equals("wlan0")) {
                    return str;
                }
            }
        } catch (SocketException socketException) {
            return null;
        }
        return null;
    }

    public static String getUserID() {
        // TODO: implement in the future
        String macAddress = getMacMoreThanM();
        if (macAddress != null) {
            return macAddress.replace(":", "_");
        } else {
            return "TEST_USERID";
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CollectorResult upload(CollectorResult result, String name, String commit) {
        return upload(result, name, commit, null);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CollectorResult upload(CollectorResult collectorResult, String name, String commit, Result contextOrAction) {
        long uploadTime = System.currentTimeMillis();
        Log.e("Upload name", name);
        Log.e("Upload commit", commit);
        Log.e("Upload file", collectorResult.getSavePath());

        File file = new File(collectorResult.getSavePath());
        File metaFile = new File(file.getAbsolutePath() + ".meta");
        TaskMetaBean meta = new TaskMetaBean(file.getName(), 0, commit, name, getUserID(), uploadTime);
        meta.setCollectorResult(JSONUtils.collectorResultToMap(collectorResult));
        meta.setContextAction(JSONUtils.resultToMap(contextOrAction));
        FileUtils.writeStringToFile(new Gson().toJson(meta), metaFile);

        uploader.pushTask(new UploadTask(file, metaFile, meta), true);
        return collectorResult;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(Collector collector, TriggerConfig triggerConfig, String name, String commit) {
        return clickTrigger.trigger(collector, triggerConfig)
                .thenApply((v) -> upload(v.get(0), name, commit));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(Collector collector, TriggerConfig triggerConfig, String name, String commit, Result contextOrAction) {
        return clickTrigger.trigger(collector, triggerConfig)
                .thenApply((v) -> upload(v.get(0), name, commit, contextOrAction));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, String name, String commit) {
        return clickTrigger.trigger(Collections.singletonList(type), triggerConfig)
                .thenApply((v) -> upload(v.get(0), name, commit));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, String name, String commit, Result contextOrAction) {
        return clickTrigger.trigger(Collections.singletonList(type), triggerConfig)
                .thenApply((v) -> upload(v.get(0), name, commit, contextOrAction));
    }
}
