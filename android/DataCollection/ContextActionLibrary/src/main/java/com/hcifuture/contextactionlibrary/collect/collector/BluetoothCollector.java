package com.hcifuture.contextactionlibrary.collect.collector;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
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

import java.util.List;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CompletableFuture;

public class BluetoothCollector extends Collector {

    private BluetoothData data;

    private BroadcastReceiver receiver;

    public BluetoothCollector(Context context, String triggerFolder) {
        super(context, triggerFolder);
        data = new BluetoothData();
    }

    private synchronized void insert(BluetoothDevice device, short rssi, boolean linked) {
        data.insert(new SingleBluetoothData(device.getName(), device.getAddress(),
                device.getBondState(), device.getType(),
                device.getBluetoothClass().getDeviceClass(),
                device.getBluetoothClass().getMajorDeviceClass(),
                rssi, linked));
    }

    @Override
    public void initialize() {
        IntentFilter bluetoothFilter = new IntentFilter(BluetoothDevice.ACTION_FOUND);
        receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (intent.getAction().equals(BluetoothDevice.ACTION_FOUND)) {
                    BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
                    short rssi = 0;
                    if (device.getBondState() != BluetoothDevice.BOND_BONDED) {
                        rssi = intent.getExtras().getShort(BluetoothDevice.EXTRA_RSSI);
                    }
                    insert(device, rssi, false);
                }
            }
        };
        mContext.registerReceiver(receiver, bluetoothFilter);
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Data> collect() {
        CompletableFuture<Data> ft = new CompletableFuture<>();
        data.clear();
        BluetoothAdapter bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
        Set<BluetoothDevice> pairedDevices = bluetoothAdapter.getBondedDevices();
        if (pairedDevices.size() > 0) {
            for (BluetoothDevice device: pairedDevices) {
                insert(device, (short)0, true);
            }
        }

        bluetoothAdapter.startDiscovery();
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                synchronized (BluetoothCollector.this) {
                    bluetoothAdapter.cancelDiscovery();
                    saver.save(data.deepClone());
                    ft.complete(data);
                }
            }
        }, 10000);
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

    }

    @Override
    public synchronized void resume() {

    }
}
