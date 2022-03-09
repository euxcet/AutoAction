package com.example.contextactionlibrary.utils;

public class Slope1C {
    private float xDelta = 0.0F;
    private float xRawLast;

    public Slope1C() {
    }

    public void init(float var1) {
        this.xRawLast = var1;
    }

    public float update(float var1, float var2) {
        var1 *= var2;
        var2 = var1 - this.xRawLast;
        this.xDelta = var2;
        this.xRawLast = var1;
        return var2;
    }
}
