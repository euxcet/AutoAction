package com.example.contextactionlibrary.utils;

public class Lowpass1C {
    private float para = 1.0F;
    private float xLast = 0.0F;

    public Lowpass1C() {
    }

    public void init(float var1) {
        this.xLast = var1;
    }

    public void setPara(float var1) {
        this.para = var1;
    }

    public float update(float var1) {
        float var2 = this.para;
        var1 = var2 * var1 + (1.0F - var2) * this.xLast;
        this.xLast = var1;
        return var1;
    }
}
