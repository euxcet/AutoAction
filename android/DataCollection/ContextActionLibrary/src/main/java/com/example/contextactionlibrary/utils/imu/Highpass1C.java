package com.example.contextactionlibrary.utils.imu;

public class Highpass1C {
    private float para = 1.0F;
    private float xLast = 0.0F;
    private float yLast = 0.0F;

    public Highpass1C() {
    }

    public void init(float var1) {
        this.xLast = var1;
        this.yLast = var1;
    }

    public void setPara(float var1) {
        this.para = var1;
    }

    public float update(float var1) {
        float var2 = this.para;
        float var3 = this.yLast;
        var2 = var2 * (var1 - this.xLast) + var3 * var2;
        this.yLast = var2;
        this.xLast = var1;
        return var2;
    }
}
