package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import com.hcifuture.contextactionlibrary.contextaction.action.MotionAction;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.contextaction.context.ConfigContext;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

import androidx.annotation.RequiresApi;

public class ConfigCollector extends BaseCollector {
    private final TriggerConfig triggerConfig;
    private long last_nonimu = 0;
    private long last_position = 0;
    private long last_scan = 0;
    private long last_audio = 0;

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

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        long current_call = action.getTimestamp();
        String commit = "";

        if (MotionAction.NEED_POSITION.equals(action.getAction())) {
            // called every 5 min at most
            if (current_call - last_position >= 5 * 60000) {
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
        long current_call = context.getTimestamp();
        String commit = "";

        if (ConfigContext.NEED_AUDIO.equals(context.getContext())) {
            last_audio = current_call;
            String name = "Event_Audio";
            triggerAndUpload(CollectorManager.CollectorType.Audio, triggerConfig, name, commit, context);
        } else if (ConfigContext.NEED_NONIMU.equals(context.getContext())) {
            // called every second at most
            if (current_call - last_nonimu >= 1000) {
                last_nonimu = current_call;
                String name = "Event_NonIMU";
                triggerAndUpload(CollectorManager.CollectorType.NonIMU, triggerConfig, name, commit, context);
            }
        } else if (ConfigContext.NEED_SCAN.equals(context.getContext())) {
            // called every minute at most
            if (current_call - last_scan >= 60000) {
                last_scan = current_call;
                String name = "Event_Scan";
                triggerAndUpload(CollectorManager.CollectorType.Bluetooth, triggerConfig, name, commit, context);
                triggerAndUpload(CollectorManager.CollectorType.Wifi, triggerConfig, name, commit, context);
            }
        }
    }
}
