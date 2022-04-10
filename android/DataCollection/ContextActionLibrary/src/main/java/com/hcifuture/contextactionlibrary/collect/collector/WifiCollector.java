package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.SystemClock;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.data.NonIMUData;
import com.hcifuture.contextactionlibrary.collect.data.SingleWifiData;
import com.hcifuture.contextactionlibrary.collect.data.WifiData;
import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;

import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class WifiCollector extends Collector {

    private WifiManager wifiManager;

    private WifiData data;

    private BroadcastReceiver receiver;

    public WifiCollector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, triggerFolder, scheduledExecutorService, futureList);
        this.data = new WifiData();
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void initialize() {
        wifiManager = (WifiManager) mContext.getSystemService(Context.WIFI_SERVICE);

        IntentFilter wifiFilter = new IntentFilter();
        wifiFilter.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);

        receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (intent.getBooleanExtra(WifiManager.EXTRA_RESULTS_UPDATED, false)) {
                    List<ScanResult> results = wifiManager.getScanResults();
                    for (ScanResult result: results) {
                        synchronized (this) {
                            data.insert(new SingleWifiData(result.SSID, result.BSSID,
                                    result.capabilities,
                                    result.level, result.frequency,
                                    result.timestamp,
                                    result.channelWidth,
                                    result.centerFreq0, result.centerFreq1, false));
                        }
                    }
                }
            }
        };

        mContext.registerReceiver(receiver, wifiFilter);
    }

    @Override
    public void setSavePath(String timestamp) {
        if (data instanceof List) {
            saver.setSavePath(timestamp + "_wifi.bin");
        }
        else
            saver.setSavePath(timestamp + "_wifi.txt");
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Void> collect() {
        CompletableFuture<Void> ft = new CompletableFuture<>();
        data.clear();
        WifiInfo info = wifiManager.getConnectionInfo();
        if (info != null) {
            data.insert(new SingleWifiData(info.getSSID(), info.getBSSID(),
                    "NULL",
                    0, info.getFrequency(),
                    SystemClock.elapsedRealtimeNanos()/1000,
                    0,
                    0, 0, true));
        }
        wifiManager.startScan();
        futureList.add(scheduledExecutorService.schedule(() -> {
            synchronized (WifiCollector.this) {
                WifiData cloneData = data.deepClone();
                saver.save(cloneData);
            }
            ft.complete(null);
        }, 10000, TimeUnit.MILLISECONDS));
        return ft;
    }

    @Override
    public void close() {
        mContext.unregisterReceiver(receiver);
    }

    @Override
    public boolean forPrediction() {
        return true;
    }

    @Override
    public synchronized Data getData() {
        Gson gson = new Gson();
        return gson.fromJson(gson.toJson(data), NonIMUData.class);
    }

    @Override
    public String getSaveFolderName() {
        return "Wifi";
    }

    @Override
    public synchronized void pause() {

    }

    @Override
    public synchronized void resume() {

    }
}
