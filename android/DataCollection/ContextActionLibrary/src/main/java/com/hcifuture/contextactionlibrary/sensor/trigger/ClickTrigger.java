package com.hcifuture.contextactionlibrary.sensor.trigger;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;

public class ClickTrigger extends Trigger {
    private final HashMap<String, List<File>> history;
    private final String saveFolder;
    private final AtomicInteger mFileID = new AtomicInteger(0);
    private final IntUnaryOperator operator = x -> (x < 999)? (x + 1) : 0;

    public ClickTrigger(Context context, CollectorManager collectorManager, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, collectorManager, scheduledExecutorService, futureList);
        this.saveFolder = context.getExternalMediaDirs()[0].getAbsolutePath() + "/Data/Click/";
        this.history = new HashMap<>();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private CompletableFuture<List<CollectorResult>> triggerCollectors(List<Collector> collectors, TriggerConfig config) {
        Log.d(TAG, "数据收集开始执行: [" + System.currentTimeMillis() + "]");
        List<CompletableFuture<CollectorResult>> fts = new ArrayList<>();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());


        for (Collector collector: collectors) {
            String name = collector.getName();
            int fileID = mFileID.getAndUpdate(operator);
            File saveFile = new File(this.saveFolder + name + "/" + name + "_" + timestamp + "_" + fileID + collector.getExt());
            synchronized (history) {
                if (!history.containsKey(name)) {
                    history.put(name, new ArrayList<>());
                }
            }
            Objects.requireNonNull(history.get(name)).add(saveFile);
            CompletableFuture<CollectorResult> ft;
            long startTimestamp = System.currentTimeMillis();
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
                    config.setAudioFilename("");
                } else { // async
                    ft = ((AsynchronousCollector)collector).getData(config)
                            .thenCompose(v -> FileUtils.writeStringToFile(v, saveFile, scheduledExecutorService, futureList));
                }
            } else { // unknown collector
                ft = CompletableFuture.completedFuture(null);
            }

            fts.add(ft.thenApply(v -> {
                v.setStartTimestamp(startTimestamp);
                v.setEndTimestamp(System.currentTimeMillis());
                v.setType(collector.getType());
                return v;
            }));
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
    public CompletableFuture<List<CollectorResult>> trigger(Collector collector, TriggerConfig config) {
        return triggerCollectors(Collections.singletonList(collector), config);
    }

}
