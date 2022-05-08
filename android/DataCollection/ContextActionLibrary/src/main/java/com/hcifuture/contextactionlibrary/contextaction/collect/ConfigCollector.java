package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import com.hcifuture.contextactionlibrary.contextaction.action.MotionAction;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.contextaction.context.ConfigContext;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.contextactionlibrary.utils.JSONUtils;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.hcifuture.shared.communicate.result.Result;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntUnaryOperator;

import androidx.annotation.RequiresApi;

public class ConfigCollector extends BaseCollector {
    private final TriggerConfig triggerConfig;
    private long last_nonimu = 0;
    private long last_position = 0;
    private LogCollector contextActionLogCollector;
    private final AtomicInteger mLogID = new AtomicInteger(0);
    private final IntUnaryOperator operator = x -> (x < 999)? (x + 1) : 0;

    @RequiresApi(api = Build.VERSION_CODES.N)
    public ConfigCollector(Context context, ScheduledExecutorService scheduledExecutorService,
                           List<ScheduledFuture<?>> futureList, RequestListener requestListener,
                           ClickTrigger clickTrigger, Uploader uploader) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
        triggerConfig = new TriggerConfig()
                .setAudioLength(5000)
                .setBluetoothScanTime(10000)
                .setGPSRequestTime(3000);
    }

    public void setContextActionLogCollector(LogCollector contextActionLogCollector) {
        this.contextActionLogCollector = contextActionLogCollector;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private int incLogID() {
        return mLogID.getAndUpdate(operator);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void recordContextAction(Result contextOrAction) {
        if (contextActionLogCollector != null) {
            String line = incLogID() + "\t" +
                    contextOrAction.getTimestamp() + "\t" +
                    contextOrAction.getKey() + "\t" +
                    contextOrAction.getReason() + "\t" +
                    JSONUtils.bundleToJSON(contextOrAction.getExtras()).toString();
            contextActionLogCollector.addLog(line);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        recordContextAction(action);
        long current_call = action.getTimestamp();
        String commit = "";

        if (MotionAction.NEED_POSITION.equals(action.getAction())) {
            // called every 10 min at most
            if (current_call - last_position >= 10 * 60000) {
                last_position = current_call;
                String name = "Event_Position";
                triggerAndUpload(CollectorManager.CollectorType.GPS, triggerConfig, name, commit, action);
                triggerAndUpload(CollectorManager.CollectorType.Location, triggerConfig, name, commit, action);
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
        recordContextAction(context);
        long current_call = context.getTimestamp();
        String commit = "";

        if (ConfigContext.NEED_AUDIO.equals(context.getContext())) {
            String name = "Event_Audio";
            triggerAndUpload(CollectorManager.CollectorType.Audio, triggerConfig, name, commit, context);
        } else if (ConfigContext.NEED_NONIMU.equals(context.getContext())) {
            if (current_call - last_nonimu >= 1000) {
                last_nonimu = current_call;
                String name = "Event_NonIMU";
                triggerAndUpload(CollectorManager.CollectorType.NonIMU, triggerConfig, name, commit, context);
            }
        } else if (ConfigContext.NEED_SCAN.equals(context.getContext())) {
            String name = "Event_Scan";
            triggerAndUpload(CollectorManager.CollectorType.Bluetooth, triggerConfig, name, commit, context);
            triggerAndUpload(CollectorManager.CollectorType.Wifi, triggerConfig, name, commit, context);
        }
    }
}
