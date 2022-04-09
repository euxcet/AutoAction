package com.hcifuture.contextactionlibrary.collect.trigger;

import android.content.Context;
import android.util.Log;

import com.hcifuture.contextactionlibrary.collect.collector.Collector;
import com.hcifuture.contextactionlibrary.collect.collector.CompleteIMUCollector;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ClickTrigger extends Trigger {

    public ClickTrigger(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, types, scheduledExecutorService, futureList);
    }

    public ClickTrigger(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }

    @Override
    public void trigger() {
        futureList.add(scheduledExecutorService.schedule(() -> {
                Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
                String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
                for (Collector collector: collectors) {
                    collector.setSavePath(timestamp);
                }
                for (Collector collector: collectors) {
                    collector.collect();
                }
                Log.d(TAG, "数据收集结束: [" + System.currentTimeMillis() + "]");
            }, 0L, TimeUnit.MILLISECONDS
        ));
    }

    @Override
    public void trigger(List<CollectorType> types) {
        futureList.add(scheduledExecutorService.schedule(() -> {
                    Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
                    String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
                    for (Collector collector: collectors) {
                        if (types.contains(collector.getType())) {
                            collector.setSavePath(timestamp);
                        }
                    }
                    for (Collector collector: collectors) {
                        if (types.contains(collector.getType())) {
                            collector.collect();
                        }
                    }
                    Log.d(TAG, "数据收集结束: [" + System.currentTimeMillis() + "]");
                }, 0L, TimeUnit.MILLISECONDS
        ));
    }

    @Override
    public void trigger(Collector collector) {
        futureList.add(scheduledExecutorService.schedule(() -> {
                    Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
                    String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
                    collector.setSavePath(timestamp);
                    collector.collect();
                    Log.d(TAG, "数据收集结束: [" + System.currentTimeMillis() + "]");
                }, 0L, TimeUnit.MILLISECONDS
        ));
    }

    @Override
    public String getName() {
        return "Data/Click";
    }

    public String getRecentIMUPath() {
        for (Collector collector: collectors) {
            if (collector.getSaveFolderName().equals("CompleteIMU")) {
                return collector.getRecentPath();
            }
        }
        return "";
    }

    public String getRecentIMUData() {
        for (Collector collector: collectors) {
            if (collector.getSaveFolderName().equals("CompleteIMU")) {
                return ((CompleteIMUCollector)collector).getRecentData();
            }
        }
        return "";
    }

    public String getTapTapPoint() {
        for (Collector collector: collectors) {
            if (collector.getSaveFolderName().equals("CompleteIMU")) {
                return ((CompleteIMUCollector)collector).getTapTapPoint();
            }
        }
        return "";
    }
}
