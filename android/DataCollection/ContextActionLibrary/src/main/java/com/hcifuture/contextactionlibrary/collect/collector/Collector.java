package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;

import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.file.Saver;

import java.util.concurrent.CompletableFuture;

public abstract class Collector {
    protected Context mContext;
    protected Saver saver;

    public Collector(Context context, String triggerFolder) {
        mContext = context;
        saver = new Saver(mContext, triggerFolder, getSaveFolderName());
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
