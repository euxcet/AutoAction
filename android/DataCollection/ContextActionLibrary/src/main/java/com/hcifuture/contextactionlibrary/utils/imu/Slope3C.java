package com.hcifuture.contextactionlibrary.utils.imu;

public class Slope3C {
    private Slope1C _slopeX = new Slope1C();
    private Slope1C _slopeY = new Slope1C();
    private Slope1C _slopeZ = new Slope1C();

    public Slope3C() {
    }

    public void init(Point3f var1) {
        this._slopeX.init(var1.x);
        this._slopeY.init(var1.y);
        this._slopeZ.init(var1.z);
    }

    public Point3f update(Point3f var1, float var2) {
        return new Point3f(this._slopeX.update(var1.x, var2), this._slopeY.update(var1.y, var2), this._slopeZ.update(var1.z, var2));
    }
}
