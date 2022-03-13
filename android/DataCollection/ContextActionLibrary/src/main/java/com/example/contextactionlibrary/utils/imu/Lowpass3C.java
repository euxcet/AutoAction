package com.example.contextactionlibrary.utils.imu;

public class Lowpass3C extends Lowpass1C {
    private Lowpass1C xLowpass = new Lowpass1C();
    private Lowpass1C yLowpass = new Lowpass1C();
    private Lowpass1C zLowpass = new Lowpass1C();

    public Lowpass3C() {
    }

    public void init(Point3f var1) {
        this.xLowpass.init(var1.x);
        this.yLowpass.init(var1.y);
        this.zLowpass.init(var1.z);
    }

    public void setPara(float var1) {
        this.xLowpass.setPara(var1);
        this.yLowpass.setPara(var1);
        this.zLowpass.setPara(var1);
    }

    public Point3f update(Point3f var1) {
        return new Point3f(this.xLowpass.update(var1.x), this.yLowpass.update(var1.y), this.zLowpass.update(var1.z));
    }
}
