package com.hcifuture.contextactionlibrary.sensor.collector.async;

public class LocationException extends Exception {

    private int code;
    public LocationException(int code, String msg) {
        super(msg);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
