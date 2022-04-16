package com.hcifuture.contextactionlibrary.sensor.collector.sync;

import android.content.Context;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.LogData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class LogCollector extends SynchronousCollector {
    private String label;
    private int historyLength;

    private LogData data;

    public LogCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, String label, int historyLength) {
        super(context, type, scheduledExecutorService, futureList);
        this.label = label;
        this.historyLength = historyLength;
        this.data = new LogData(historyLength);
    }

    @Override
    public void initialize() {
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<Void> collect(TriggerConfig config) {
        CompletableFuture<Void> ft = new CompletableFuture<>();
        saver.save(data.getString()).whenComplete((v, t) -> ft.complete(null));
        return ft;
    }
     */

    public void addLog(String log) {
        data.addLog(log);
    }

    public void eraseLog() {
        data.eraseLog();
    }

    @Override
    public void close() {
        data.clear();
    }

    @Override
    public Data getData(TriggerConfig config) {
        return data.clone();
    }

    @Override
    public String getDataString(TriggerConfig config) {
        return data.getString();
    }

    @Override
    public void pause() {

    }

    @Override
    public void resume() {

    }

    @Override
    public String getName() {
        return "Log";
    }

    @Override
    public String getExt() {
        return ".txt";
    }
}
