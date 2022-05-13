package com.hcifuture.contextactionlibrary.utils;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;

import java.io.File;
import java.util.concurrent.CompletableFuture;

public class SaveTask {
    private CollectorResult result;
    private File saveFile;
    private CompletableFuture<CollectorResult> future;
    private int type;

    public SaveTask(File saveFile, CollectorResult result, CompletableFuture<CollectorResult> future, int type) {
        this.saveFile = saveFile;
        this.result = result;
        this.future = future;
        this.type = type;
    }

    public void setResult(CollectorResult result) {
        this.result = result;
    }

    public CollectorResult getResult() {
        return result;
    }

    public File getSaveFile() {
        return saveFile;
    }

    public void setSaveFile(File saveFile) {
        this.saveFile = saveFile;
    }

    public CompletableFuture<CollectorResult> getFuture() {
        return future;
    }

    public void setFuture(CompletableFuture<CollectorResult> future) {
        this.future = future;
    }

    public void setType(int type) {
        this.type = type;
    }

    public int getType() {
        return type;
    }
}
