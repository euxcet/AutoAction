package com.hcifuture.contextactionlibrary.sensor.uploader;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class UploadTask implements Comparable<UploadTask> {
    private static final int DEFAULT_REMAINING_RETRIES = 5;
    private File file;
    private File metaFile;
    private List<TaskMetaBean> meta;
    private int remainingRetries;

    private long expectedUploadTime;

    public UploadTask(File file, File metaFile, TaskMetaBean meta, int remainingRetries) {
        this.file = file;
        this.metaFile = metaFile;
        this.meta = Collections.singletonList(meta);
        this.remainingRetries = remainingRetries;
        this.expectedUploadTime = System.currentTimeMillis();
    }

    public UploadTask(File file, File metaFile, List<TaskMetaBean> meta, int remainingRetries) {
        this.file = file;
        this.metaFile = metaFile;
        this.meta = meta;
        this.remainingRetries = remainingRetries;
        this.expectedUploadTime = System.currentTimeMillis();
    }

    public UploadTask(File file, File metaFile, TaskMetaBean meta) {
        this(file, metaFile, meta, DEFAULT_REMAINING_RETRIES);
    }

    public UploadTask(File file, File metaFile, List<TaskMetaBean> meta) {
        this(file, metaFile, meta, DEFAULT_REMAINING_RETRIES);
    }

    @Override
    public int compareTo(UploadTask uploadTask) {
        return Long.compare(this.expectedUploadTime, uploadTask.expectedUploadTime);
    }

    public File getFile() {
        return file;
    }

    public int getRemainingRetries() {
        return remainingRetries;
    }

    public void setFile(File file) {
        this.file = file;
    }

    public void setRemainingRetries(int remainingRetries) {
        this.remainingRetries = remainingRetries;
    }

    public static int getDefaultRemainingRetries() {
        return DEFAULT_REMAINING_RETRIES;
    }

    public void setExpectedUploadTime(long expectedUploadTime) {
        this.expectedUploadTime = expectedUploadTime;
    }

    public long getExpectedUploadTime() {
        return expectedUploadTime;
    }

    public void setMeta(List<TaskMetaBean> meta) {
        this.meta = meta;
    }

    public List<TaskMetaBean> getMeta() {
        return meta;
    }

    public File getMetaFile() {
        return metaFile;
    }

    public void setMetaFile(File metaFile) {
        this.metaFile = metaFile;
    }
}
