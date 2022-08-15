package com.hcifuture.datacollection.action;

public class ObjectDescriptor {
    private float[] imageFeature;

    public ObjectDescriptor(float[] imageFeature) {
        this.imageFeature = imageFeature;
    }

    public float[] getImageFeature() {
        return imageFeature;
    }

    public void setImageFeature(float[] imageFeature) {
        this.imageFeature = imageFeature;
    }
}
