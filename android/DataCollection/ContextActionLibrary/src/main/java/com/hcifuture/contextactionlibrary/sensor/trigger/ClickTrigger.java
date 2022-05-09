package com.hcifuture.contextactionlibrary.sensor.trigger;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.collector.async.AsynchronousCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.AudioCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.IMUCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.SynchronousCollector;
import com.hcifuture.contextactionlibrary.utils.FileUtils;
import com.hcifuture.contextactionlibrary.utils.JSONUtils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;

public class ClickTrigger extends Trigger {
    private final String saveFolder;
    private final AtomicInteger mTriggerIDCounter = new AtomicInteger(0);
    private final IntUnaryOperator operator = x -> (x < 999)? (x + 1) : 0;
    private LogCollector triggerLogCollector;
    private final Gson gson;

    public ClickTrigger(Context context, CollectorManager collectorManager, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, collectorManager, scheduledExecutorService, futureList);
        this.saveFolder = context.getExternalMediaDirs()[0].getAbsolutePath() + "/Data/Click/";
        this.gson = new Gson();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private int incTriggerID() {
        return mTriggerIDCounter.getAndUpdate(operator);
    }

    public void setTriggerLogCollector(LogCollector triggerLogCollector) {
        this.triggerLogCollector = triggerLogCollector;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private CompletableFuture<List<CollectorResult>> triggerCollectors(List<Collector> collectors, TriggerConfig config) {
        List<CompletableFuture<CollectorResult>> fts = new ArrayList<>();
        for (Collector collector : collectors) {
            fts.add(trigger(collector, config)
                    .exceptionally(ex -> null));
        }
        CompletableFuture<Void> allFutures = CompletableFuture.allOf(fts.toArray(new CompletableFuture[0]));
        return allFutures.thenApply(v ->
                fts.stream().map(ft -> {
                    try {
                        return ft.get();
                    } catch (ExecutionException | InterruptedException e) {
                        e.printStackTrace();
                    }
                    return null;
                }).collect(Collectors.toList())
        );
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<List<CollectorResult>> trigger(TriggerConfig config) {
        return triggerCollectors(collectorManager.getCollectors(), config);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<List<CollectorResult>> trigger(List<CollectorManager.CollectorType> types, TriggerConfig config) {
        return triggerCollectors(collectorManager.getCollectors().stream().filter(
                (collector) -> types.contains(collector.getType()) && (collector.getType() != CollectorManager.CollectorType.Log)
        ).collect(Collectors.toList()), config);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> trigger(Collector collector, TriggerConfig config) {
        String dateTime = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String collectorName = collector.getName();
        int triggerID = incTriggerID();
        String filename = collectorName + "_" + dateTime + "_" + triggerID + collector.getExt();
        File saveFile = new File(this.saveFolder + collectorName + "/" + filename);
        CompletableFuture<CollectorResult> ft;
        long startTimestamp = System.currentTimeMillis();

        Log.e(TAG, "数据收集开始执行: [" + startTimestamp + "] " + collectorName + " " + saveFile.getAbsolutePath());

        if (collector instanceof SynchronousCollector) { // sync
            ft = FileUtils.writeStringToFile(((SynchronousCollector)collector).getData(config),
                    saveFile, scheduledExecutorService, futureList);
        } else if (collector instanceof AsynchronousCollector) {
            if (collector instanceof IMUCollector) { // imu
                ft = ((IMUCollector) collector).getData(config)
                        .thenCompose(v -> FileUtils.writeIMUDataToFile(v, saveFile, scheduledExecutorService, futureList));
            } else if (collector instanceof AudioCollector) { // audio
                config.setAudioFilename(saveFile.getAbsolutePath());
                ft = ((AsynchronousCollector)collector).getData(config);
                config.setAudioFilename(null);
            } else { // async
                ft = ((AsynchronousCollector)collector).getData(config)
                        .thenCompose(v -> FileUtils.writeStringToFile(v, saveFile, scheduledExecutorService, futureList));
            }
        } else { // unknown collector
            ft = new CompletableFuture<>();
            ft.completeExceptionally(new Exception("Unknown collector"));
        }

        return ft.whenComplete((v, ex) -> {
            if (ex == null) {
                v.setStartTimestamp(startTimestamp);
                v.setEndTimestamp(System.currentTimeMillis());
                v.setType(collector.getType());
                if (collector.getType() == CollectorManager.CollectorType.Log) {
                    v.setName(collectorName);
                }
            }
            /*
                Logging:
                    timestamp | triggerID | type | name | code | filename | triggerConfig | collectorResult
                code:
                    0: successful complete
                    1: exception
            */
            if (triggerLogCollector != null) {
                String line = System.currentTimeMillis() + "\t"
                        + triggerID + "\t"
                        + collector.getType() + "\t"
                        + collectorName + "\t"
                        + ((ex == null) ? 0 : 1) + "\t"
                        + filename + "\t"
                        + gson.toJson(config) + "\t"
                        + ((ex == null) ? gson.toJson(JSONUtils.collectorResultToMap(v)) : ex.toString());
                triggerLogCollector.addLog(line);
            }
        });
    }
}
