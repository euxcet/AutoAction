package com.hcifuture.contextactionlibrary.collect.trigger;

import android.content.Context;

import com.hcifuture.contextactionlibrary.collect.collector.Collector;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class TimerTrigger extends Trigger {

    public TimerTrigger(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, types, scheduledExecutorService, futureList);
    }

    public TimerTrigger(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }

    @Override
    public void trigger() {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
                for (Collector collector: collectors) {
                    collector.setSavePath(timestamp);
                    collector.collect();
                }
            }
        }, 5000, 600000);
    }

    @Override
    public String getName() {
        return "Data/Timer";
    }
}
