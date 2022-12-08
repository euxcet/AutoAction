package com.hcifuture.datacollection.data;

import android.graphics.Bitmap;

public class CameraCaptureResult {
    private Bitmap bitmap;
    private float[] feature;

    public CameraCaptureResult(Bitmap bitmap, float[] feature) {
        this.bitmap = bitmap;
        this.feature = feature;
    }

    public Bitmap getBitmap() {
        return bitmap;
    }

    public float[] getFeature() {
        return feature;
    }

    public void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
    }

    public void setFeature(float[] feature) {
        this.feature = feature;
    }
}
