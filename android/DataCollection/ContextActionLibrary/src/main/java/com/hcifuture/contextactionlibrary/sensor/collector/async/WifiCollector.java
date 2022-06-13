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

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorException;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.SingleWifiData;
import com.hcifuture.contextactionlibrary.sensor.data.WifiData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.status.Heart;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

public class WifiCollector extends AsynchronousCollector {

    private final WifiData data;

    private WifiManager wifiManager;
    private BroadcastReceiver receiver;
    private IntentFilter wifiFilter;

    private final AtomicBoolean isCollecting;
    private final AtomicBoolean isScanning;
    private final Object lockScan;
    private final Object lockStopScan;
    private CompletableFuture<CollectorResult> mFt;
    private long resultTimestamp = 0;

    /*
      Error code:
        0: no error
        1: Cannot start Wifi scan
        2: Wifi scan results not updated
        3: Concurrent task of Wifi scanning
        4: Unknown collecting exception
        5: Unknown exception when getting scan results
        6: Scan timeout
        7: Unknown exception when ft.get()
        8: 5 & 6, or 5 & 7
        9: Invalid Wifi scan timeout
     */

    public WifiCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        isCollecting = new AtomicBoolean(false);
        isScanning = new AtomicBoolean(false);
        lockScan = new Object();
        lockStopScan = new Object();
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

                // Is scanning
                synchronized (lockScan) {
                    if (isCollecting.get() && isScanning.get()) {
                        synchronized (lockStopScan) {
                            if (mFt != null && !mFt.isDone()) {
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
                                    isScanning.set(false);
                                }
                            }
                        }
                    }
                }
            }
        };

        mContext.registerReceiver(receiver, wifiFilter, null, handler);
        isRegistered.set(true);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        Heart.getInstance().newSensorGetEvent(getName(), System.currentTimeMillis());
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();

        if (config.getWifiScanTimeout() <= 0) {
            ft.completeExceptionally(new CollectorException(9, "Invalid Wifi scan timeout: " + config.getWifiScanTimeout()));
        } else if (isCollecting.compareAndSet(false, true)) {
            try {
                notifyWake();
                setBasicInfo();
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
                synchronized (lockScan) {
                    mFt = ft;
                    if (!wifiManager.startScan()) {
                        insertScanResults();
                        setCollectData(result);
                        result.setErrorCode(1);
                        result.setErrorReason("Cannot start Wifi scan");
                        ft.complete(result);
                        isCollecting.set(false);
                        isScanning.set(false);
                    } else {
                        isScanning.set(true);
                        // start a new thread to check Bluetooth scan timeout
                        futureList.add(scheduledExecutorService.schedule(() -> {
                            try {
                                ft.get(config.getWifiScanTimeout(), TimeUnit.MILLISECONDS);
                            } catch (Exception e) {
                                synchronized (lockStopScan) {
                                    if (!ft.isDone()) {
                                        try {
                                            if (e instanceof TimeoutException) {
                                                result.setErrorCode(6);
                                                result.setErrorReason("Wifi scan timeout: longer than " + config.getWifiScanTimeout());
                                            } else {
                                                result.setErrorCode(7);
                                                result.setErrorReason(e.toString());
                                            }
                                            insertScanResults();
                                        } catch (Exception e1) {
                                            result.setErrorCode(8);
                                            result.setErrorReason(result.getErrorReason() + " | " + e1);
                                        } finally {
                                            setCollectData(result);
                                            ft.complete(result);
                                            isCollecting.set(false);
                                            isScanning.set(false);
                                        }
                                    }
                                }
                            }
                        }, 0, TimeUnit.MILLISECONDS));
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                ft.completeExceptionally(new CollectorException(4, e));
                isCollecting.set(false);
                isScanning.set(false);
            }
        } else {
            ft.completeExceptionally(new CollectorException(3, "Concurrent task of Wifi scanning"));
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
            mContext.registerReceiver(receiver, wifiFilter, null, handler);
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

    private void setBasicInfo() {
        if (wifiManager != null) {
            data.setState(wifiManager.getWifiState());
        }
    }
}
