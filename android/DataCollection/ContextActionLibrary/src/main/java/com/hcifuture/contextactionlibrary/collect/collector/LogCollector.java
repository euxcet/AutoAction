package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.data.LogData;
import com.hcifuture.contextactionlibrary.collect.file.Saver;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;
import com.hcifuture.contextactionlibrary.collect.trigger.TriggerConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class LogCollector extends Collector {
    private String label;
    private int historyLength;

    private LogData data;

    public LogCollector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, String label, int historyLength) {
        super(context, type, triggerFolder, scheduledExecutorService, futureList);
        this.label = label;
        this.historyLength = historyLength;
        this.data = new LogData(historyLength);
        this.saver = new Saver(mContext, triggerFolder, getSaveFolderName(), scheduledExecutorService, futureList);
    }

    @Override
    public void initialize() {
    }

    @Override
    public void setSavePath(String timestamp) {
        saver.setSavePath(timestamp + "_log.txt");
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<Void> collect(TriggerConfig config) {
        CompletableFuture<Void> ft = new CompletableFuture<>();
        saver.save(data.getString()).whenComplete((v, t) -> ft.complete(null));
        return ft;
    }

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
    public boolean forPrediction() {
        return false;
    }

    @Override
    public Data getData() {
        return data;
    }

    @Override
    public String getSaveFolderName() {
        return "Log_" + label;
    }

    @Override
    public void pause() {

    }

    @Override
    public void resume() {

    }
}
