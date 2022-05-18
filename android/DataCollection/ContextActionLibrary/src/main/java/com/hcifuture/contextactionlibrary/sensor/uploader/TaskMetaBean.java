package com.hcifuture.contextactionlibrary.sensor.uploader;

import java.util.Map;

public class TaskMetaBean {
    private String file;
    private int fileType;
    private String commit;
    private String name;
    private String userId;
    private long timestamp;
    private Map<String, Object> CollectorResult;
    private Map<String, Object> ContextAction;

    public TaskMetaBean(String file, int fileType, String commit, String name, String userId, long timestamp) {
        this.file = file;
        this.fileType = fileType;
        this.commit = commit;
        this.name = name;
        this.userId = userId;
        this.timestamp = timestamp;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public void setFileType(int fileType) {
        this.fileType = fileType;
    }

    public void setFile(String file) {
        this.file = file;
    }

    public void setCommit(String commit) {
        this.commit = commit;
    }

    public String getUserId() {
        return userId;
    }

    public String getCommit() {
        return commit;
    }

    public int getFileType() {
        return fileType;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public String getFile() {
        return file;
    }

    public Map<String, Object> getCollectorResult() {
        return CollectorResult;
    }

    public void setCollectorResult(Map<String, Object> collectorResult) {
        this.CollectorResult = collectorResult;
    }

    public Map<String, Object> getContextAction() {
        return ContextAction;
    }

    public void setContextAction(Map<String, Object> contextAction) {
        this.ContextAction = contextAction;
    }
}
