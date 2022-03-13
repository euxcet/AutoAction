package com.example.contextactionlibrary.utils.imu;

public class Sample3C {
    public Point3f point;
    public long t;

    public Sample3C(float var1, float var2, float var3, long var4) {
        Point3f var6 = new Point3f(0.0F, 0.0F, 0.0F);
        this.point = var6;
        var6.x = var1;
        var6.y = var2;
        var6.z = var3;
        this.t = var4;
    }
}
