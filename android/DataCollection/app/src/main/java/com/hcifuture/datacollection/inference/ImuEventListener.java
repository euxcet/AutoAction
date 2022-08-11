package com.hcifuture.datacollection.inference;

public interface ImuEventListener {
    void onStatus(String status);
    void onAction(String action);
}
