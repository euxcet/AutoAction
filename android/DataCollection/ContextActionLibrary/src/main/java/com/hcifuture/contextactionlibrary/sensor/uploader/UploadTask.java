package com.hcifuture.contextactionlibrary.sensor.uploader;

import java.io.File;

public class UploadTask implements Comparable<UploadTask> {
    private static final int DEFAULT_REMAINING_RETRIES = 5;
    private File file;
    private int fileType;
    private String commit;
    private String name;
    private String userId;
    private long timestamp;
    private int remainingRetries;

    private long expectedUploadTime;

    public UploadTask(File file, int fileType,
                      String commit, String name, String userId,
                      long timestamp, int remainingRetries) {
        this.file = file;
        this.fileType = fileType;
        this.commit = commit;
        this.name = name;
        this.userId = userId;
        this.timestamp = timestamp;
        this.remainingRetries = remainingRetries;
        this.expectedUploadTime = System.currentTimeMillis();
    }

    @Override
    public int compareTo(UploadTask uploadTask) {
        return Long.compare(this.expectedUploadTime, uploadTask.expectedUploadTime);
    }

    public UploadTask(File file, int fileType,
                      String commit, String name, String userId,
                      long timestamp) {
        this(file, fileType, commit, name, userId, timestamp, DEFAULT_REMAINING_RETRIES);
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public File getFile() {
        return file;
    }

    public int getFileType() {
        return fileType;
    }

    public int getRemainingRetries() {
        return remainingRetries;
    }

    public String getCommit() {
        return commit;
    }

    public String getUserId() {
        return userId;
    }

    public void setCommit(String commit) {
        this.commit = commit;
    }

    public void setFile(File file) {
        this.file = file;
    }

    public void setFileType(int fileType) {
        this.fileType = fileType;
    }

    public void setRemainingRetries(int remainingRetries) {
        this.remainingRetries = remainingRetries;
    }

    public void setUserId(String userId) {
        this.userId = userId;
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
}
