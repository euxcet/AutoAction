package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.BluetoothData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleBluetoothData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class BluetoothCollector extends AsynchronousCollector {

    private final BluetoothData data;

    private BroadcastReceiver receiver;
    private IntentFilter bluetoothFilter;
        
    private BluetoothAdapter bluetoothAdapter;
    private BluetoothManager bluetoothManager;
    private BluetoothLeScanner bluetoothLeScanner;
    private ScanCallback leScanCallback;

    private final AtomicBoolean isCollecting;

    /*
      Error code:
        0: No error
        1: Cannot start Bluetooth discovery
        2: Cannot get BluetoothLeScanner
        3: Both 1 & 2
        4: Cannot cancel Bluetooth discovery
        5: Invalid Bluetooth scan time
        6: Concurrent task of Bluetooth scanning
        7: Unknown collecting exception
        8: Unknown exception when stopping scan
        9: Gson serialization error
     */

    public BluetoothCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        data = new BluetoothData();
        isCollecting = new AtomicBoolean(false);
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @SuppressLint("MissingPermission")
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();

        if (config.getBluetoothScanTime() <= 0) {
            result.setErrorCode(5);
            result.setErrorReason("Invalid Bluetooth scan time: " + config.getBluetoothScanTime());
//            ft.complete(result);
            ft.completeExceptionally(new Exception("Invalid Bluetooth scan time: " + config.getBluetoothScanTime()));
        } else if (isCollecting.compareAndSet(false, true)) {
            try {
                notifyWake();
                setBasicInfo();
                data.clear();

                // scan bonded (paired) devices
                Set<BluetoothDevice> pairedDevices = bluetoothAdapter.getBondedDevices();
                if (pairedDevices.size() > 0) {
                    for (BluetoothDevice device: pairedDevices) {
                        insert(device, isConnected(device, false), null, null);
                    }
                }

                // scan connected BLE devices
                List<BluetoothDevice> connectedDevices = bluetoothManager.getConnectedDevices(BluetoothProfile.GATT);
                for (BluetoothDevice device : connectedDevices) {
                    insert(device, isConnected(device, true), null, null);
                }

                int errorCode = 0;
                String errorReason = "";

                // start classic bluetooth scanning
                if (!bluetoothAdapter.startDiscovery()) {
                    bluetoothAdapter.cancelDiscovery();
                    errorCode += 1;
                    errorReason += "Cannot start Bluetooth discovery";
                }

                // start BLE scanning
                bluetoothLeScanner = bluetoothAdapter.getBluetoothLeScanner();
                if (bluetoothLeScanner == null) {
                    errorCode += 2;
                    errorReason += " | Cannot get BluetoothLeScanner";
                } else {
                    bluetoothLeScanner.startScan(leScanCallback);
                }

                if (errorCode != 0) {
                    setCollectData(result);
                    result.setErrorCode(errorCode);
                    result.setErrorReason(errorReason);
                    ft.complete(result);
                    isCollecting.set(false);
                } else {
                    // Stops scanning after given time
                    futureList.add(scheduledExecutorService.schedule(() -> {
                        try {
                            boolean success = stopScan();
                            if (!success) {
                                result.setErrorCode(4);
                                result.setErrorReason("Cannot cancel Bluetooth discovery");
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                            result.setErrorCode(8);
                            result.setErrorReason(e.toString());
                        } finally {
                            try {
                                setCollectData(result);
                            } catch (Exception e) {
                                result.setErrorCode(9);
                                result.setErrorReason(e.toString());
                            } finally {
                                ft.complete(result);
                                isCollecting.set(false);
                            }
                        }
                    }, config.getBluetoothScanTime(), TimeUnit.MILLISECONDS));
                }
            } catch (Exception e) {
                e.printStackTrace();
                stopScan();
                result.setErrorCode(7);
                result.setErrorReason(e.toString());
                ft.complete(result);
                isCollecting.set(false);
            }
        } else {
            result.setErrorCode(6);
            result.setErrorReason("Concurrent task of Bluetooth scanning");
//            ft.complete(result);
            ft.completeExceptionally(new Exception("Concurrent task of Bluetooth scanning"));
        }

        return ft;
    }

    @SuppressLint("MissingPermission")
    private boolean stopScan() {
        boolean ret = true;
        if (bluetoothLeScanner != null) {
            bluetoothLeScanner.stopScan(leScanCallback);
        }
        if (bluetoothAdapter != null) {
            ret = bluetoothAdapter.cancelDiscovery();
        }
        return ret;
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return getData(config).thenApply((d) -> d == null ? null : gson.toJson(d, BluetoothData.class));
    }
     */

    @SuppressLint("MissingPermission")
    private void insert(BluetoothDevice device, boolean linked, ScanResult scanResult, Bundle intentExtra) {
        SingleBluetoothData singleBluetoothData = new SingleBluetoothData(device, linked, scanResult, intentExtra);
        data.insert(singleBluetoothData);
    }

    @Override
    public void initialize() {
        // initializes Bluetooth manager and adapter
        bluetoothManager = (BluetoothManager) mContext.getSystemService(Context.BLUETOOTH_SERVICE);
        bluetoothAdapter = bluetoothManager.getAdapter();

        // set classic bluetooth scan callback
        bluetoothFilter = new IntentFilter(BluetoothDevice.ACTION_FOUND);
        receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (intent.getAction().equals(BluetoothDevice.ACTION_FOUND)) {
                    BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
                    insert(device, isConnected(device, false), null, intent.getExtras());
                }
            }
        };

        mContext.registerReceiver(receiver, bluetoothFilter);
        isRegistered.set(true);

        // ref: https://developer.android.com/guide/topics/connectivity/bluetooth-le#find
        // set BLE scan callback
        leScanCallback = new ScanCallback() {
            @Override
            public void onScanResult (int callbackType, ScanResult result) {
                handler.post(() -> {
                    BluetoothDevice device = result.getDevice();
                    insert(device, isConnected(device, false), result, null);
                });
            }
        };
    }

    // ref: https://stackoverflow.com/a/58882930/11854304
    @SuppressLint("MissingPermission")
    private boolean isConnected(BluetoothDevice device, boolean defaultValue) {
        try {
            Method m = device.getClass().getMethod("isConnected", (Class[]) null);
            return (boolean) m.invoke(device, (Object[]) null);
        } catch (Exception e) {
//            throw new IllegalStateException(e);
            try {
                return bluetoothManager.getConnectionState(device, BluetoothProfile.GATT) == BluetoothProfile.STATE_CONNECTED;
            } catch (Exception e1) {
                return defaultValue;
            }
        }
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @SuppressLint("MissingPermission")
    @Override
    public synchronized CompletableFuture<Void> collect(TriggerConfig config) {
    }
     */

    @SuppressLint("MissingPermission")
    @Override
    public void close() {
        stopScan();
        if (isRegistered.get() && receiver != null) {
            mContext.unregisterReceiver(receiver);
            isRegistered.set(false);
        }
    }


    @Override
    public void pause() {
        stopScan();
        if (isRegistered.get() && receiver != null) {
            mContext.unregisterReceiver(receiver);
            isRegistered.set(false);
        }
    }

    @Override
    public void resume() {
        if (!isRegistered.get() && receiver != null && bluetoothFilter != null) {
            mContext.registerReceiver(receiver, bluetoothFilter);
            isRegistered.set(true);
        }
    }

    @Override
    public String getName() {
        return "Bluetooth";
    }

    @Override
    public String getExt() {
        return ".json";
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    private void setCollectData(CollectorResult result) {
        result.setData(data.deepClone());
        result.setDataString(gson.toJson(result.getData(), BluetoothData.class));
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @SuppressLint("MissingPermission")
    private void setBasicInfo() {
        if (bluetoothAdapter != null) {
            data.setAddress(bluetoothAdapter.getAddress());
            data.setLeMaximumAdvertisingDataLength(bluetoothAdapter.getLeMaximumAdvertisingDataLength());
            data.setName(bluetoothAdapter.getName());
            data.setProfileConnectionState_A2DP(bluetoothAdapter.getProfileConnectionState(BluetoothProfile.A2DP));
            data.setProfileConnectionState_HEADSET(bluetoothAdapter.getProfileConnectionState(BluetoothProfile.HEADSET));
            data.setScanMode(bluetoothAdapter.getScanMode());
            data.setState(bluetoothAdapter.getState());
            data.setDiscovering(bluetoothAdapter.isDiscovering());
            data.setLe2MPhySupported(bluetoothAdapter.isLe2MPhySupported());
            data.setLeCodedPhySupported(bluetoothAdapter.isLeCodedPhySupported());
            data.setLeExtendedAdvertisingSupported(bluetoothAdapter.isLeExtendedAdvertisingSupported());
            data.setLePeriodicAdvertisingSupported(bluetoothAdapter.isLePeriodicAdvertisingSupported());
            data.setMultipleAdvertisementSupported(bluetoothAdapter.isMultipleAdvertisementSupported());
            data.setOffloadedFilteringSupported(bluetoothAdapter.isOffloadedFilteringSupported());
            data.setOffloadedScanBatchingSupported(bluetoothAdapter.isOffloadedScanBatchingSupported());
        }
    }
}
