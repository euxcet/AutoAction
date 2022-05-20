package com.hcifuture.contextactionlibrary.sensor.collector;

public class CollectorException extends Exception {

    private final int code;
    public CollectorException(int code, String msg) {
        super(msg);
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    @Override
    public String toString() {
        return "CollectorException(" + code + "): " + getLocalizedMessage();
    }
}
