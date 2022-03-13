package com.example.contextactionlibrary.collect.trigger;

import android.content.Context;
import android.util.Log;

import com.example.contextactionlibrary.collect.collector.Collector;
import com.example.contextactionlibrary.collect.collector.CompleteIMUCollector;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ClickTrigger extends Trigger {

    private ThreadPoolExecutor threadPoolExecutor;
    private TriggerTask triggerTask;

    public ClickTrigger(Context context, List<CollectorType> types) {
        super(context, types);
    }

    public ClickTrigger(Context context, CollectorType type) {
        super(context, type);
        triggerTask = new TriggerTask();
        this.threadPoolExecutor = new ThreadPoolExecutor(
                1,
                2,
                2,
                TimeUnit.MINUTES,
                new PriorityBlockingQueue<>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.AbortPolicy());
    }

    @Override
    public void trigger() {
        if (threadPoolExecutor != null) {
            threadPoolExecutor.execute(triggerTask);
        }
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
}
