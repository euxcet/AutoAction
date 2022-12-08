package com.hcifuture.datacollection.inference;

public class InferenceResult {
    public int classId;
    public String className;
    public float prob;

    public InferenceResult(int classId, String className, float prob) {
        this.classId = classId;
        this.className = className;
        this.prob = prob;
    }
}
