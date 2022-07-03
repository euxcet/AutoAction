package com.hcifuture.datacollection.utils.bean;

/**
 * Stores some logs in a training process.
 */
public class TrainLogBean {
    private int epoch;
    private float loss;
    private float train_acc;
    private float val_acc;

    public float getLoss() {
        return loss;
    }

    public float getTrain_acc() {
        return train_acc;
    }

    public float getVal_acc() {
        return val_acc;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    public void setLoss(float loss) {
        this.loss = loss;
    }

    public void setTrain_acc(float train_acc) {
        this.train_acc = train_acc;
    }

    public void setVal_acc(float val_acc) {
        this.val_acc = val_acc;
    }
}