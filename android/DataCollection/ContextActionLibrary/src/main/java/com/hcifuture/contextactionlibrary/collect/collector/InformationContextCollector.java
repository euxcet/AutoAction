package com.hcifuture.contextactionlibrary.collect.collector;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.collect.data.InformationContextData;
import com.hcifuture.contextactionlibrary.collect.data.BluetoothData;
import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.data.InformationData;
import com.hcifuture.contextactionlibrary.collect.data.SingleBluetoothData;

import java.util.Date;
import java.util.List;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class InformationContextCollector extends Collector {

    private InformationContextData data;


    public InformationContextCollector(Context context, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, triggerFolder, scheduledExecutorService, futureList);
        data = new InformationContextData();
    }

    private synchronized void insert(String task, String type, Date date) {
        data.insert(new InformationData(task,type,date));
    }

    @Override
    public void initialize() {
    }

    @Override
    public void setSavePath(String timestamp) {
        if (data instanceof List) {
            saver.setSavePath(timestamp + "_informationContext.bin");
        }
        else {
            saver.setSavePath(timestamp + "_informationContext.txt");
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Data> collect() {
        CompletableFuture<Data> ft = new CompletableFuture<>();

        return ft;
    }

    @Override
    public void close() {
    }

    @Override
    public boolean forPrediction() {
        return true;
    }

    @Override
    public synchronized Data getData() {
        Gson gson = new Gson();
        return gson.fromJson(gson.toJson(data), BluetoothData.class);
    }

    @Override
    public String getSaveFolderName() {
        return "InformationContext";
    }

    @Override
    public synchronized void pause() {
    }

    @Override
    public synchronized void resume() {
    }
}
