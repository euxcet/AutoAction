package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

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

    public BaseCollector(Context context, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, RequestListener requestListener, ClickTrigger clickTrigger) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.requestListener = requestListener;
        this.clickTrigger = clickTrigger;
        this.futureList = futureList;
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
    public CompletableFuture<CollectorResult> upload(CollectorResult result, String name, String commit) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        try {
            long uploadTime = System.currentTimeMillis();
            String newCommit = "Type: " + ((result.getType() == null)? "Unknown" : result.getType()) + "\n" +
                    "Start: " + result.getStartTimestamp() + "\n" +
                    "End: " + result.getEndTimestamp() + "\n" +
                    "Upload: " + uploadTime + "\n" + commit;
            Log.e("Upload name", name);
            Log.e("Upload commit", newCommit);
            Log.e("Upload file", result.getSavePath());
            File file = new File(result.getSavePath());
            NetworkUtils.uploadCollectedData(mContext,
                    file,
                    0,
                    name,
                    getUserID(),
                    uploadTime,
                    newCommit,
                    new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                            Log.e("Upload response", "Success!");
                            ft.complete(result);
                        }

                        @Override
                        public void onError(Response<String> response) {
                            ft.completeExceptionally(response.getException());
                            super.onError(response);
                        }
                    });
        } catch (Exception e) {
            e.printStackTrace();
            ft.completeExceptionally(e);
        }
        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> upload(CollectorResult result, String name, String commit, long timestamp) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        try {
            Log.e("Upload name", name);
            Log.e("Upload commit", commit);
            Log.e("Upload file", result.getSavePath());
            File file = new File(result.getSavePath());
            NetworkUtils.uploadCollectedData(mContext,
                    file,
                    0,
                    name,
                    getUserID(),
                    timestamp,
                    commit,
                    new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                            Log.e("Upload response", "Success!");
                            ft.complete(result);
                        }

                        @Override
                        public void onError(Response<String> response) {
                            ft.completeExceptionally(response.getException());
                            super.onError(response);
                        }
                    });
        } catch (Exception e) {
            e.printStackTrace();
            ft.completeExceptionally(e);
        }
        return ft;
    }


    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(Collector collector, TriggerConfig triggerConfig, String name, String commit) {
        return clickTrigger.trigger(collector, triggerConfig)
                .thenCompose((v) -> upload(v.get(0), name, commit));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(Collector collector, TriggerConfig triggerConfig, String name, String commit, long time) {
        return clickTrigger.trigger(collector, triggerConfig)
                .thenCompose((v) -> upload(v.get(0), name, commit, time));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> triggerAndUpload(CollectorManager.CollectorType type, TriggerConfig triggerConfig, String name, String commit) {
        return clickTrigger.trigger(Collections.singletonList(type), triggerConfig)
                .thenCompose((v) -> upload(v.get(0), name, commit));
    }
}
