package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;

import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.file.Saver;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;
import com.hcifuture.contextactionlibrary.collect.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class Collector {
    protected Context mContext;
    protected Saver saver;
    protected ScheduledExecutorService scheduledExecutorService;
    protected List<ScheduledFuture<?>> futureList;
    protected Trigger.CollectorType type;

    public Collector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.type = type;
        this.saver = new Saver(mContext, triggerFolder, getSaveFolderName(), scheduledExecutorService, futureList);
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        initialize();
    }

    public abstract void initialize();

    public abstract void setSavePath(String timestamp);

    public abstract CompletableFuture<Void> collect(TriggerConfig config);

    public abstract void close();

    public abstract boolean forPrediction();

    public abstract Data getData();

    public abstract String getSaveFolderName();

    public abstract void pause();

    public abstract void resume();

    public String getRecentPath() {
        return saver.getSavePath();
    }

    public Trigger.CollectorType getType() {
        return type;
    }

    public void cleanData() {
        saver.deleteFolderFile(saver.getSaveFolder(), false);
    }
}
