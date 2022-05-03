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
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.BluetoothData;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.SingleBluetoothData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import org.checkerframework.checker.units.qual.A;

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
    private ScanCallback leScanCallback;

    private final AtomicBoolean isCollecting;

    public BluetoothCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        data = new BluetoothData();
        isCollecting = new AtomicBoolean(false);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @SuppressLint("MissingPermission")
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        if (config.getBluetoothScanTime() <= 0) {
            ft.completeExceptionally(new Exception("Invalid Bluetooth scan time: " + config.getBluetoothScanTime()));
            return ft;
        }

        if (!isCollecting.get()) {
            isCollecting.set(true);
            try {
                data.clear();

                // scan bonded (paired) devices
                Set<BluetoothDevice> pairedDevices = bluetoothAdapter.getBondedDevices();
                if (pairedDevices.size() > 0) {
                    for (BluetoothDevice device: pairedDevices) {
                        insert(device, (short)0, isConnected(device), null, null);
                    }
                }

                // scan connected BLE devices
                List<BluetoothDevice> connectedDevices = bluetoothManager.getConnectedDevices(BluetoothProfile.GATT);
                for (BluetoothDevice device : connectedDevices) {
                    insert(device, (short)0, isConnected(device), null, null);
                }

                int errorCode = 0;
                String errorReason = "";

                // start classic bluetooth scanning
                if (!bluetoothAdapter.startDiscovery()) {
                    errorCode += 1;
                    errorReason += "Cannot start Bluetooth discovery";
                    isCollecting.set(false);
                }

                // start BLE scanning
                BluetoothLeScanner bluetoothLeScanner = bluetoothAdapter.getBluetoothLeScanner();
                if (bluetoothLeScanner == null) {
                    errorCode += 2;
                    errorReason += " | Cannot get BluetoothLeScanner";
                } else {
                    bluetoothLeScanner.startScan(leScanCallback);
                }

                if (errorCode != 0) {
                    CollectorResult result = getResult();
                    result.setErrorCode(errorCode);
                    result.setErrorReason(errorReason);
                    ft.complete(result);
                    isCollecting.set(false);
                } else {
                    // Stops scanning after given time
                    futureList.add(scheduledExecutorService.schedule(() -> {
                        try {
                            bluetoothLeScanner.stopScan(leScanCallback);
                            bluetoothAdapter.cancelDiscovery();
                            ft.complete(getResult());
                        } catch (Exception e) {
                            e.printStackTrace();
                            ft.completeExceptionally(e);
                        } finally {
                            isCollecting.set(false);
                        }
                    }, config.getBluetoothScanTime(), TimeUnit.MILLISECONDS));
                }
            } catch (Exception e) {
                e.printStackTrace();
                ft.completeExceptionally(e);
                isCollecting.set(false);
            }
        } else {
            ft.completeExceptionally(new Exception("Another task of Bluetooth scanning is taking place!"));
        }

        return ft;
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return getData(config).thenApply((d) -> d == null ? null : gson.toJson(d, BluetoothData.class));
    }
     */

    @SuppressLint("MissingPermission")
    private synchronized void insert(BluetoothDevice device, short rssi, boolean linked, String scanResult, String intentExtra) {
        data.insert(new SingleBluetoothData(device.getName(), device.getAddress(),
                device.getBondState(), device.getType(),
                device.getBluetoothClass().getDeviceClass(),
                device.getBluetoothClass().getMajorDeviceClass(),
                rssi, linked, scanResult, intentExtra));
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
                    short rssi = intent.getShortExtra(BluetoothDevice.EXTRA_RSSI, (short) 0);
                    insert(device, rssi, isConnected(device), null, intent.getExtras().toString());
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
                BluetoothDevice device = result.getDevice();
                int rssi = result.getRssi();
                insert(device, (short) rssi, isConnected(device), result.toString(), null);
            }
        };
    }

    // ref: https://stackoverflow.com/a/58882930/11854304
    public static boolean isConnected(BluetoothDevice device) {
        try {
            Method m = device.getClass().getMethod("isConnected", (Class[]) null);
            return (boolean) m.invoke(device, (Object[]) null);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @SuppressLint("MissingPermission")
    @Override
    public synchronized CompletableFuture<Void> collect(TriggerConfig config) {
    }
     */

    @Override
    public void close() {
        if (isRegistered.get() && receiver != null) {
            mContext.unregisterReceiver(receiver);
            isRegistered.set(false);
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    private CollectorResult getResult() {
        CollectorResult result = new CollectorResult();
        result.setData(data.deepClone());
        result.setDataString(gson.toJson(result.getData(), BluetoothData.class));
        return result;
    }
}
