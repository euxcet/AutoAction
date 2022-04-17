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

    private BluetoothData data;

    private BroadcastReceiver receiver;
    private IntentFilter bluetoothFilter;
        
    private BluetoothAdapter bluetoothAdapter;
    private BluetoothManager bluetoothManager;
    private ScanCallback leScanCallback;

    public BluetoothCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        data = new BluetoothData();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @SuppressLint("MissingPermission")
    @Override
    public CompletableFuture<Data> getData(TriggerConfig config) {
        CompletableFuture<Data> ft = new CompletableFuture<>();
        if (config.getBluetoothScanTime() == 0) {
            ft.complete(null);
            return ft;
        }

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

        // start classic bluetooth scanning
        bluetoothAdapter.startDiscovery();
        // start BLE scanning
        BluetoothLeScanner bluetoothLeScanner = bluetoothAdapter.getBluetoothLeScanner();
        if (bluetoothLeScanner != null) {
            bluetoothLeScanner.startScan(leScanCallback);
        }

        // Stops scanning after 10 seconds
        futureList.add(scheduledExecutorService.schedule(() -> {
            try {
                synchronized (BluetoothCollector.this) {
                    if (bluetoothLeScanner != null) {
                        bluetoothLeScanner.stopScan(leScanCallback);
                    }
                    bluetoothAdapter.cancelDiscovery();
                    ft.complete(data.deepClone());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, config.getBluetoothScanTime(), TimeUnit.MILLISECONDS));
        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return getData(config).thenApply((d) -> d == null ? null : gson.toJson(d, BluetoothData.class));
    }

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
        return ".txt";
    }
}
