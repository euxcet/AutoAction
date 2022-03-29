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

    // private ThreadPoolExecutor threadPoolExecutor;
    // private TriggerTask triggerTask;

    public ClickTrigger(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, types, scheduledExecutorService, futureList);
        /*
        triggerTask = new TriggerTask();
        this.threadPoolExecutor = new ThreadPoolExecutor(
                1,
                2,
                2,
                TimeUnit.MINUTES,
                new PriorityBlockingQueue<>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.AbortPolicy());
         */
    }

    public ClickTrigger(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        /*
        triggerTask = new TriggerTask();
        this.threadPoolExecutor = new ThreadPoolExecutor(
                1,
                2,
                2,
                TimeUnit.MINUTES,
                new PriorityBlockingQueue<>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.AbortPolicy());
         */
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
        /*
        if (threadPoolExecutor != null) {
            threadPoolExecutor.execute(triggerTask);
        }
         */
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

    /*
    private class TriggerTask implements Runnable, Comparable {

        @Override
        public int compareTo(Object o) {
            return 0;
        }

        @Override
        public void run() {
            synchronized (ClickTrigger.class) {
                Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
                String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
                for (Collector collector: collectors) {
                    collector.setSavePath(timestamp);
                }
                for (Collector collector: collectors) {
                    collector.collect();
                }
                Log.d(TAG, "数据收集结束: [" + System.currentTimeMillis() + "]");
            }
        }
    }

     */
}
