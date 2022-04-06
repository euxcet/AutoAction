package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;

import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.file.Saver;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class Collector {
    protected Context mContext;
    protected Saver saver;
    protected ScheduledExecutorService scheduledExecutorService;
    protected List<ScheduledFuture<?>> futureList;

    public Collector(Context context, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.saver = new Saver(mContext, triggerFolder, getSaveFolderName());
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        initialize();
    }

    public abstract void initialize();

    public abstract void setSavePath(String timestamp);

    public abstract CompletableFuture<Data> collect();

    public abstract void close();

    public abstract boolean forPrediction();

    public abstract Data getData();

    public abstract String getSaveFolderName();

    public abstract void pause();

    public abstract void resume();

    public String getRecentPath() {
        return saver.getSavePath();
    }

    public void cleanData() {
        saver.deleteFolderFile(saver.getSaveFolder(), false);
    }
}
