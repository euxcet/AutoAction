package com.hcifuture.contextactionlibrary.collect.trigger;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.collect.collector.Collector;
import com.hcifuture.contextactionlibrary.collect.collector.CompleteIMUCollector;
import com.hcifuture.contextactionlibrary.collect.data.Data;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class ClickTrigger extends Trigger {

    public ClickTrigger(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, types, scheduledExecutorService, futureList);
    }

    public ClickTrigger(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private CompletableFuture<Void> triggerCollectors(List<Collector> collectors) {
        Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
        List<CompletableFuture<Void>> fts = new ArrayList<>();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        for (Collector collector : collectors) {
            collector.setSavePath(timestamp);
        }
        for (Collector collector : collectors) {
            fts.add(collector.collect());
        }
        return CompletableFuture.allOf(fts.toArray(new CompletableFuture[0]));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<Void> trigger() {
        return triggerCollectors(collectors);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<Void> trigger(List<CollectorType> types) {
        return triggerCollectors(collectors.stream().filter(types::contains).collect(Collectors.toList()));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<Void> trigger(Collector collector) {
        return triggerCollectors(Collections.singletonList(collector));
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<Void> triggerShortIMU(int head, int tail) {
        Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
        List<CompletableFuture<Void>> fts = new ArrayList<>();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        for (Collector collector : collectors) {
            if (collector.getSaveFolderName().equals("CompleteIMU")) {
                collector.setSavePath(timestamp);
            }
        }
        for (Collector collector : collectors) {
            if (collector.getSaveFolderName().equals("CompleteIMU")) {
                fts.add(((CompleteIMUCollector)collector).collectShort(head, tail));
            }
        }
        return CompletableFuture.allOf(fts.toArray(new CompletableFuture[0]));
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
