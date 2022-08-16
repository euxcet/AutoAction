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

    public float distance(float[] frame) {
        float result = 0.0f;
        for (int i = 0; i < frame.length; i++) {
            result += (frame[i] - imageFeature[i]) * (frame[i] - imageFeature[i]);
        }
        return result / frame.length;
    }
}
