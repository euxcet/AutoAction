package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.bluetooth.BluetoothDevice;
import android.bluetooth.le.ScanResult;
import android.content.Context;
import android.os.Bundle;
import android.os.ParcelUuid;
import android.util.SparseArray;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.utils.GsonUtils;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class AsynchronousCollector extends Collector {
    protected Gson gson;

    public AsynchronousCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        gson = new GsonBuilder().disableHtmlEscaping()
                .registerTypeAdapter(Bundle.class, GsonUtils.bundleSerializer)
                .registerTypeAdapter(ScanResult.class, GsonUtils.scanResultSerializer)
                .registerTypeAdapter(new TypeToken<SparseArray<byte[]>>(){}.getType(), new GsonUtils.SparseArraySerializer<byte[]>())
                .registerTypeAdapter(BluetoothDevice.class, GsonUtils.bluetoothDeviceSerializer)
                .registerTypeAdapter(ParcelUuid.class, GsonUtils.parcelUuidSerializer)
                .create();
    }

    public abstract CompletableFuture<CollectorResult> getData(TriggerConfig config);

    // public abstract CompletableFuture<String> getDataString(TriggerConfig config);
}
