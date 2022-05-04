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

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.SingleWifiData;
import com.hcifuture.contextactionlibrary.sensor.data.WifiData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public class WifiCollector extends AsynchronousCollector {

    private final WifiData data;

    private WifiManager wifiManager;
    private BroadcastReceiver receiver;
    private IntentFilter wifiFilter;

    private final AtomicBoolean isCollecting;
    private CompletableFuture<CollectorResult> mFt;
    private long startScanTimestamp = 0;
    private long resultTimestamp = 0;

    /*
      Error code:
        0: no error
        1: Cannot start Wifi scan
        2: Wifi scan results not updated
        3: Concurrent task of Wifi scanning
        4: Unknown collecting exception
        5: Unknown exception when getting scan results
     */

    public WifiCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        isCollecting = new AtomicBoolean(false);
        this.data = new WifiData();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void initialize() {
        wifiManager = (WifiManager) mContext.getApplicationContext().getSystemService(Context.WIFI_SERVICE);

        wifiFilter = new IntentFilter();
        wifiFilter.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);

        receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                boolean updated = intent.getBooleanExtra(WifiManager.EXTRA_RESULTS_UPDATED, false);
                if (updated) {
                    resultTimestamp = System.currentTimeMillis();
                }

                // Is recording
                // onReceive may be called before startScanTimestamp (due to system Wifi scan)
                if (isCollecting.get() && System.currentTimeMillis() >= startScanTimestamp) {
                    CollectorResult result = new CollectorResult();
                    try {
                        if (!updated) {
                            result.setErrorCode(2);
                            result.setErrorReason("Wifi scan results not updated");
                        }
                        insertScanResults();
                    } catch (Exception e) {
                        e.printStackTrace();
                        result.setErrorCode(5);
                        result.setErrorReason(e.toString());
                    } finally {
                        setCollectData(result);
                        mFt.complete(result);
                        isCollecting.set(false);
                    }
                }
            }
        };

        mContext.registerReceiver(receiver, wifiFilter);
        isRegistered.set(true);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();

        if (!isCollecting.get()) {
            isCollecting.set(true);
            try {
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
                startScanTimestamp = System.currentTimeMillis();
                if (!wifiManager.startScan()) {
                    insertScanResults();
                    setCollectData(result);
                    result.setErrorCode(1);
                    result.setErrorReason("Cannot start Wifi scan");
                    ft.complete(result);
                    isCollecting.set(false);
                } else {
                    mFt = ft;
                }
            } catch (Exception e) {
                e.printStackTrace();
                result.setErrorCode(4);
                result.setErrorReason(e.toString());
                ft.complete(result);
                isCollecting.set(false);
            }
        } else {
            result.setErrorCode(3);
            result.setErrorReason("Concurrent task of Wifi scanning");
            ft.complete(result);
        }

        return ft;
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return getData(config).thenApply((d) -> d == null ? null : gson.toJson(d, WifiData.class));
    }
     */

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
        return ".json";
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    void insertScanResults() {
        List<ScanResult> results = wifiManager.getScanResults();
        for (ScanResult result : results) {
            data.insert(new SingleWifiData(result.SSID, result.BSSID,
                    result.capabilities,
                    result.level, result.frequency,
                    result.timestamp,
                    result.channelWidth,
                    result.centerFreq0, result.centerFreq1, false));
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void setCollectData(CollectorResult collectorResult) {
        collectorResult.setData(data.deepClone());
        collectorResult.setDataString(gson.toJson(collectorResult.getData(), WifiData.class));
        collectorResult.getExtras().putLong("ResultTimestamp", resultTimestamp);
    }
}
