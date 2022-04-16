package com.hcifuture.contextactionlibrary.sensor.trigger;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.AsynchronousCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.AudioCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.async.IMUCollector;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.SynchronousCollector;
import com.hcifuture.contextactionlibrary.sensor.data.IMUData;
import com.hcifuture.contextactionlibrary.utils.FileUtils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.stream.Collectors;

public class ClickTrigger extends Trigger {
    private HashMap<String, List<File>> history;
    private String saveFolder;
    public ClickTrigger(Context context, CollectorManager collectorManager, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, collectorManager, scheduledExecutorService, futureList);
        this.saveFolder = context.getExternalMediaDirs()[0].getAbsolutePath() + "/Data/Click/";
        this.history = new HashMap<>();
    }

    /*
    public ClickTrigger(Context context, List<CollectorType> types, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, types, scheduledExecutorService, futureList);
    }

    public ClickTrigger(Context context, CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }
     */

    @RequiresApi(api = Build.VERSION_CODES.N)
    private CompletableFuture<List<String>> triggerCollectors(List<Collector> collectors, TriggerConfig config) {
        Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
        List<CompletableFuture<String>> fts = new ArrayList<>();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());


        for (Collector collector: collectors) {
            String name = collector.getName();
            File saveFile = new File(this.saveFolder + name + "/" + timestamp + collector.getExt());
            if (!history.containsKey(name)) {
                history.put(name, new ArrayList<>());
            }
            Objects.requireNonNull(history.get(name)).add(saveFile);
            Log.e("TEST", saveFile.getAbsolutePath());
            if (collector instanceof SynchronousCollector) { // sync
                fts.add(FileUtils.writeStringToFile(((SynchronousCollector)collector).getDataString(config),
                        saveFile, scheduledExecutorService, futureList));
            } else if (collector instanceof AsynchronousCollector) {
                if (collector instanceof IMUCollector) { // imu
                    fts.add(((IMUCollector) collector).getData(config).thenCompose((v) ->
                            FileUtils.writeIMUDataToFile((IMUData)v, saveFile, scheduledExecutorService, futureList)));
                } else if (collector instanceof AudioCollector) {
                    config.setAudioFilename(saveFile.getAbsolutePath());
                    fts.add(((AsynchronousCollector)collector).getData(config).thenApply((v) -> null));
                    config.setAudioFilename("");
                } else { // async
                    fts.add(((AsynchronousCollector)collector).getDataString(config).thenCompose((v) ->
                            FileUtils.writeStringToFile(v, saveFile, scheduledExecutorService, futureList)));
                }
            }
        }

        CompletableFuture<Void> allFutures = CompletableFuture.allOf(fts.toArray(new CompletableFuture[0]));

        return allFutures.thenApply(v -> fts.stream().map(ft -> {
            try {
                return ft.get();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
            return null;
        }).collect(Collectors.toList()));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<List<String>> trigger(TriggerConfig config) {
        return triggerCollectors(collectorManager.getCollectors(), config);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<List<String>> trigger(List<CollectorManager.CollectorType> types, TriggerConfig config) {
        return triggerCollectors(collectorManager.getCollectors().stream().filter(
                (collector) -> types.contains(collector.getType()) && (collector.getType() != CollectorManager.CollectorType.Log)
        ).collect(Collectors.toList()), config);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<List<String>> trigger(Collector collector, TriggerConfig config) {
        return triggerCollectors(Collections.singletonList(collector), config);
    }

    /*
    @Override
    public String getName() {
        return "Data/Click";
    }
     */

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<Void> triggerShortIMU(int head, int tail) {
        Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
        List<CompletableFuture<Void>> fts = new ArrayList<>();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        for (Collector collector : collectors) {
            if (collector.getType() == CollectorType.CompleteIMU) {
                collector.setSavePath(timestamp);
            }
        }
        for (Collector collector : collectors) {
            if (collector.getType() == CollectorType.CompleteIMU) {
                fts.add(((CompleteIMUCollector)collector).collectShort(head, tail));
            }
        }
        return CompletableFuture.allOf(fts.toArray(new CompletableFuture[0]));
    }
     */
}
