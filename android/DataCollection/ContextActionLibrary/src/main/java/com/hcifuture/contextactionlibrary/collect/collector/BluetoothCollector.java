package com.hcifuture.contextactionlibrary.collect.collector;

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

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.collect.data.BluetoothData;
import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.data.SingleBluetoothData;
import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class BluetoothCollector extends Collector {

    private BluetoothData data;

    private BroadcastReceiver receiver;
    private IntentFilter bluetoothFilter;
        
    private BluetoothAdapter bluetoothAdapter;
    private BluetoothManager bluetoothManager;
    private ScanCallback leScanCallback;

    public BluetoothCollector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, triggerFolder, scheduledExecutorService, futureList);
        data = new BluetoothData();
    }

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

    @Override
    public void setSavePath(String timestamp) {
        if (data instanceof List) {
            saver.setSavePath(timestamp + "_bluetooth.bin");
        }
        else {
            saver.setSavePath(timestamp + "_bluetooth.txt");
        }
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Data> collect() {
        CompletableFuture<Data> ft = new CompletableFuture<>();
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
        bluetoothLeScanner.startScan(leScanCallback);

        // Stops scanning after 10 seconds
        futureList.add(scheduledExecutorService.schedule(() -> {
            synchronized (BluetoothCollector.this) {
                bluetoothLeScanner.stopScan(leScanCallback);
                bluetoothAdapter.cancelDiscovery();
                saver.save(data.deepClone());
                ft.complete(data);
            }
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
        return gson.fromJson(gson.toJson(data), BluetoothData.class);
    }

    @Override
    public String getSaveFolderName() {
        return "Bluetooth";
    }

    @Override
    public synchronized void pause() {
        mContext.unregisterReceiver(receiver);
    }

    @Override
    public synchronized void resume() {
        mContext.registerReceiver(receiver, bluetoothFilter);
    }
}
