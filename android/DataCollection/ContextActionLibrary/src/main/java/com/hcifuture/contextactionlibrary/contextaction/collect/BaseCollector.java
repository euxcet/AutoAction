package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.collect.collector.Collector;
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
    public CompletableFuture<Void> upload(String filePath, String name, String commit) {
        CompletableFuture<Void> ft = new CompletableFuture<>();
        try {
            Log.e("Upload name", name);
            Log.e("Upload commit", commit);
            Log.e("Upload file", filePath);
            File file = new File(filePath);
            NetworkUtils.uploadCollectedData(mContext,
                    file,
                    0,
                    name,
                    getUserID(),
                    System.currentTimeMillis(),
                    commit,
                    new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                            Log.e("Upload response", "Success!");
                            ft.complete(null);
                        }
                    });
        } catch (Exception e) {
            e.printStackTrace();
        }
        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<Void> triggerAndUpload(Collector collector, TriggerConfig triggerConfig, String name, String commit) {
        CompletableFuture<Void> ft = new CompletableFuture<>();
        clickTrigger.trigger(collector, triggerConfig).whenComplete((msg, ex) -> {
            upload(collector.getRecentPath(), name, commit).whenComplete((v, t) -> ft.complete(null));
        });
        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<Void> triggerAndUpload(Trigger.CollectorType type, TriggerConfig triggerConfig, String name, String commit) {
        CompletableFuture<Void> ft = new CompletableFuture<>();
        clickTrigger.trigger(Collections.singletonList(type), triggerConfig).whenComplete((msg, ex) -> {
            upload(clickTrigger.getRecentPath(type), name, commit).whenComplete((v, t) -> ft.complete(null));
        });
        return ft;
    }
}
