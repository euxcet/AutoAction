package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.SingleWifiData;
import com.hcifuture.contextactionlibrary.sensor.data.WifiData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class WifiCollector extends AsynchronousCollector {

    private WifiManager wifiManager;

    private WifiData data;

    private BroadcastReceiver receiver;

    private IntentFilter wifiFilter;

    public WifiCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        this.data = new WifiData();
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void initialize() {
        wifiManager = (WifiManager) mContext.getApplicationContext().getSystemService(Context.WIFI_SERVICE);

        wifiFilter = new IntentFilter();
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
        isRegistered.set(true);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<Data> getData(TriggerConfig config) {
        CompletableFuture<Data> ft = new CompletableFuture<>();
        if (config.getWifiScanTime() == 0) {
            ft.complete(null);
            return ft;
        }

        data.clear();
        WifiInfo info = wifiManager.getConnectionInfo();
        if (info != null && info.getBSSID() != null) {
            data.insert(new SingleWifiData(info.getSSID(), info.getBSSID(),
                    null,
                    0, info.getFrequency(),
                    SystemClock.elapsedRealtimeNanos()/1000,
                    0,
                    0, 0, true));
        }
        wifiManager.startScan();
        futureList.add(scheduledExecutorService.schedule(() -> {
            try {
                synchronized (WifiCollector.this) {
                    ft.complete(data.deepClone());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, config.getWifiScanTime(), TimeUnit.MILLISECONDS));
        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return getData(config).thenApply((d) -> d == null ? null : gson.toJson(d, WifiData.class));
    }

    @Override
    public void close() {
        if (isRegistered.get() && receiver != null) {
            mContext.unregisterReceiver(receiver);
        }
    }

    @Override
    public void pause() {
        if (isRegistered.get() && receiver != null) {
            mContext.unregisterReceiver(receiver);
            isRegistered.set(false);
        }
    }

    @Override
    public void resume() {
        if (!isRegistered.get() && receiver != null && wifiFilter != null) {
            mContext.registerReceiver(receiver, wifiFilter);
            isRegistered.set(true);
        }
    }

    @Override
    public String getName() {
        return "Wifi";
    }

    @Override
    public String getExt() {
        return ".txt";
    }
}
