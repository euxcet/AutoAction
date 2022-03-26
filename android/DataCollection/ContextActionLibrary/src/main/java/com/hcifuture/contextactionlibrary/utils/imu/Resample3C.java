package com.hcifuture.contextactionlibrary.utils.imu;

public class Resample3C extends Resample1C {
    private float yRawLast;
    private float yResampledThis;
    private float zRawLast;
    private float zResampledThis;

    public Resample3C() {
    }

    public Sample3C getResults() {
        return new Sample3C(super.xResampledThis, this.yResampledThis, this.zResampledThis, super.tResampledLast);
    }

    public void init(float var1, float var2, float var3, long var4, long var6) {
        this.init(var1, var4, var6);
        this.yRawLast = var2;
        this.zRawLast = var3;
        this.yResampledThis = var2;
        this.zResampledThis = var3;
    }

    public boolean update(float var1, float var2, float var3, long var4) {
        long var6 = super.tRawLast;
        boolean var8;
        if (var4 == var6) {
            var8 = false;
        } else {
            long var9 = super.tInterval;
            long var11 = var9;
            if (var9 <= 0L) {
                var11 = var4 - var6;
            }

            var9 = var11 + super.tResampledLast;
            if (var4 < var9) {
                super.tRawLast = var4;
                super.xRawLast = var1;
                this.yRawLast = var2;
                this.zRawLast = var3;
                var8 = false;
            } else {
                var11 = super.tRawLast;
                float var13 = (float)(var9 - var11) / (float)(var4 - var11);
                float var14 = super.xRawLast;
                super.xResampledThis = var14 + (var1 - var14) * var13;
                var14 = this.yRawLast;
                this.yResampledThis = var14 + (var2 - var14) * var13;
                var14 = this.zRawLast;
                this.zResampledThis = var13 * (var3 - var14) + var14;
                super.tResampledLast = var9;
                if (var11 < var9) {
                    super.tRawLast = var4;
                    super.xRawLast = var1;
                    this.yRawLast = var2;
                    this.zRawLast = var3;
                }

                var8 = true;
            }
        }

        return var8;
    }
}
