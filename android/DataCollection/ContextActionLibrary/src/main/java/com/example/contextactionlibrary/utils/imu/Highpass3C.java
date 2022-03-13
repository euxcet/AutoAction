package com.example.contextactionlibrary.utils.imu;

public class Highpass3C {
    private Highpass1C xHighpass = new Highpass1C();
    private Highpass1C yHighpass = new Highpass1C();
    private Highpass1C zHighpass = new Highpass1C();

    public Highpass3C() {
    }

    public void init(Point3f var1) {
        this.xHighpass.init(var1.x);
        this.yHighpass.init(var1.y);
        this.zHighpass.init(var1.z);
    }

    public void setPara(float var1) {
        this.xHighpass.setPara(var1);
        this.yHighpass.setPara(var1);
        this.zHighpass.setPara(var1);
    }

    public Point3f update(Point3f var1) {
        return new Point3f(this.xHighpass.update(var1.x), this.yHighpass.update(var1.y), this.zHighpass.update(var1.z));
    }
}
