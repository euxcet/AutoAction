package com.hcifuture.contextactionlibrary.sensor.collector;

import com.hcifuture.contextactionlibrary.sensor.data.Data;

public class CollectorResult {
    private String savePath;
    private String dataString;
    private Data data;
    private int logLength;

    public int getLogLength() {
        return logLength;
    }

    public String getSavePath() {
        return savePath;
    }

    public Data getData() {
        return data;
    }

    public String getDataString() {
        return dataString;
    }

    public CollectorResult setLogLength(int logLength) {
        this.logLength = logLength;
        return this;
    }

    public CollectorResult setSavePath(String savePath) {
        this.savePath = savePath;
        return this;
    }

    public CollectorResult setData(Data data) {
        this.data = data;
        return this;
    }

    public CollectorResult setDataString(String dataString) {
        this.dataString = dataString;
        return this;
    }
}
