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
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.io.File;
import java.io.IOException;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Response;

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
    public CollectorResult upload(CollectorResult result, String name, String appendCommit) {
        try {
            long uploadTime = System.currentTimeMillis();
            String newCommit = "Type: " + ((result.getType() == null)? "Unknown" : result.getType()) + "\n" +
                    "Start: " + result.getStartTimestamp() + "\n" +
                    "End: " + result.getEndTimestamp() + "\n" +
                    "Upload: " + uploadTime + "\n" + appendCommit;
            Log.e("Upload name", name);
            Log.e("Upload commit", newCommit);
            Log.e("Upload file", result.getSavePath());
            File file = new File(result.getSavePath());
            File metaFile = new File(file.getAbsolutePath() + ".meta");
            TaskMetaBean meta = new TaskMetaBean(file.getName(), 0, newCommit, name, getUserID(), uploadTime);
            FileUtils.writeStringToFile(new Gson().toJson(meta), metaFile);
            uploader.pushTask(new UploadTask(file, metaFile, meta), true);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(Collector collector, TriggerConfig triggerConfig, String name, String appendCommit) {
        return clickTrigger.trigger(collector, triggerConfig)
                .thenApply((v) -> upload(v.get(0), name, appendCommit));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, String name, String appendCommit) {
        return clickTrigger.trigger(Collections.singletonList(type), triggerConfig)
                .thenApply((v) -> upload(v.get(0), name, appendCommit));
    }
}
