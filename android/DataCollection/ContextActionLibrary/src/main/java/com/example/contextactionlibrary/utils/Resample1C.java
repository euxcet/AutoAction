package com.example.contextactionlibrary.utils;


public class Resample1C {
    protected long tInterval = 0L;
    protected long tRawLast;
    protected long tResampledLast;
    protected float xRawLast;
    protected float xResampledThis = 0.0F;

    public Resample1C() {
    }

    public long getInterval() {
        return this.tInterval;
    }

    public void init(float var1, long var2, long var4) {
        this.xRawLast = var1;
        this.tRawLast = var2;
        this.xResampledThis = var1;
        this.tResampledLast = var2;
        this.tInterval = var4;
    }

    public void setSyncTime(long var1) {
        this.tResampledLast = var1;
    }
}
