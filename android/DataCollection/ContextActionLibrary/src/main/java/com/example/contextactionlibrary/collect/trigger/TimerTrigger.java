package com.example.contextactionlibrary.collect.trigger;

import android.content.Context;

import com.example.contextactionlibrary.collect.collector.Collector;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

public class TimerTrigger extends Trigger {

    public TimerTrigger(Context context, List<CollectorType> types) {
        super(context, types);
    }

    public TimerTrigger(Context context, CollectorType type) {
        super(context, type);
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
