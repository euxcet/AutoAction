package com.hcifuture.contextactionlibrary.utils;

import android.annotation.SuppressLint;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.le.ScanRecord;
import android.bluetooth.le.ScanResult;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelUuid;
import android.util.Log;
import android.util.SparseArray;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import androidx.annotation.RequiresApi;

public class GsonUtils {

    public static final String TAG = "GsonUtils";

    public static final String TO_STRING = "gsonUtilsToString";

    public static JsonElement wrapObjRecursive(Object obj, JsonSerializationContext context) {
        if (obj == null) {
            return JsonNull.INSTANCE;
        } else if (obj.getClass().isArray() &&
                obj.getClass().getComponentType() != null &&
                !obj.getClass().getComponentType().isPrimitive()) {
//            Log.e(TAG, "wrapObjRecursive: start array conversion");
            JsonArray jsonArray = new JsonArray();
            Object[] arr = (Object[]) obj;
            for (Object tmp : arr) {
                jsonArray.add(wrapObjRecursive(tmp, context));
            }
//            Log.e(TAG, "wrapObjRecursive: finish array conversion");
            return jsonArray;
        } else {
            JsonElement jsonElement = (obj instanceof JsonElement)? (JsonElement) obj : context.serialize(obj);
            if (jsonElement.isJsonObject() && ((JsonObject) jsonElement).entrySet().isEmpty()) {
//                Log.e(TAG, "wrapObjRecursive: toString");
                return new JsonPrimitive(obj.toString());
            }
            return jsonElement;
        }
    }

    public static final JsonSerializer<Bundle> bundleSerializer = new JsonSerializer<Bundle>() {
        @Override
        public JsonElement serialize(Bundle src, Type typeOfSrc, JsonSerializationContext context) {
            if (src == null) {
                return JsonNull.INSTANCE;
            } else {
                JsonObject jsonObject = new JsonObject();
                jsonObject.addProperty(TO_STRING, src.toString());
                for (String key : src.keySet()) {
                    Object value = src.get(key);
                    JsonElement jsonElement = wrapObjRecursive(value, context);
                    jsonObject.add(key, jsonElement);
                }
                return jsonObject;
            }
        }
    };

    public static final JsonSerializer<ScanResult> scanResultSerializer = new JsonSerializer<ScanResult>() {
        @RequiresApi(api = Build.VERSION_CODES.O)
        @Override
        public JsonElement serialize(ScanResult src, Type typeOfSrc, JsonSerializationContext context) {
            if (src == null) {
                return JsonNull.INSTANCE;
            } else {
                JsonObject jsonObject = new JsonObject();
                jsonObject.addProperty(TO_STRING, src.toString());

                jsonObject.add("device", context.serialize(src.getDevice(), BluetoothDevice.class));
                jsonObject.addProperty("rssi", src.getRssi());
                jsonObject.addProperty("timestampNanos", src.getTimestampNanos());
                jsonObject.addProperty("primaryPhy", src.getPrimaryPhy());
                jsonObject.addProperty("secondaryPhy", src.getSecondaryPhy());
                jsonObject.addProperty("advertisingSid", src.getAdvertisingSid());
                jsonObject.addProperty("txPower", src.getTxPower());
                jsonObject.addProperty("periodicAdvertisingInterval", src.getPeriodicAdvertisingInterval());
                ScanRecord scanRecord = src.getScanRecord();
                JsonObject json_scanRecord = new JsonObject();
                json_scanRecord.addProperty("advertiseFlags", scanRecord.getAdvertiseFlags());
                json_scanRecord.add("serviceUuids", context.serialize(scanRecord.getServiceUuids(), new TypeToken<List<ParcelUuid>>(){}.getType()));
                json_scanRecord.add("manufacturerSpecificData", context.serialize(scanRecord.getManufacturerSpecificData(), new TypeToken<SparseArray<byte[]>>(){}.getType()));
                json_scanRecord.add("serviceData", context.serialize(scanRecord.getServiceData(), new TypeToken<Map<ParcelUuid, byte[]>>(){}.getType()));
                json_scanRecord.addProperty("txPowerLevel", scanRecord.getTxPowerLevel());
                json_scanRecord.addProperty("deviceName", scanRecord.getDeviceName());
                json_scanRecord.add("bytes", context.serialize(scanRecord.getBytes(), byte[].class));
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    json_scanRecord.add("serviceSolicitationUuids", context.serialize(scanRecord.getServiceSolicitationUuids(), new TypeToken<List<ParcelUuid>>(){}.getType()));
                }
                jsonObject.add("scanRecord", json_scanRecord);

                return jsonObject;
            }
        }
    };

    public static class SparseArraySerializer<T> implements JsonSerializer<SparseArray<T>> {
        @Override
        public JsonElement serialize(SparseArray<T> src, Type typeOfSrc, JsonSerializationContext context) {
            if (src == null) {
                return JsonNull.INSTANCE;
            } else {
                JsonObject jsonObject = new JsonObject();
                for (int index = 0; index < src.size(); index++) {
                    jsonObject.add(Integer.toString(src.keyAt(index)), context.serialize(src.valueAt(index)));
                }
                return jsonObject;
            }
        }
    }

    public static final JsonSerializer<BluetoothDevice> bluetoothDeviceSerializer = new JsonSerializer<BluetoothDevice>() {
        @SuppressLint("MissingPermission")
        @Override
        public JsonElement serialize(BluetoothDevice src, Type typeOfSrc, JsonSerializationContext context) {
            if (src == null) {
                return JsonNull.INSTANCE;
            } else {
                JsonObject jsonObject = new JsonObject();
                jsonObject.addProperty("name", src.getName());
                jsonObject.addProperty("address", src.getAddress());
                jsonObject.addProperty("bondState", src.getBondState());
                jsonObject.addProperty("type", src.getType());
                jsonObject.addProperty("deviceClass", src.getBluetoothClass().getDeviceClass());
                jsonObject.addProperty("majorDeviceClass", src.getBluetoothClass().getMajorDeviceClass());
                jsonObject.add("uuids", context.serialize(src.getUuids(), ParcelUuid[].class));
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    jsonObject.addProperty("alias", src.getAlias());
                }
                return jsonObject;
            }
        }
    };

    public static final JsonSerializer<ParcelUuid> parcelUuidSerializer = new JsonSerializer<ParcelUuid>() {
        @SuppressLint("MissingPermission")
        @Override
        public JsonElement serialize(ParcelUuid src, Type typeOfSrc, JsonSerializationContext context) {
            if (src == null) {
                return JsonNull.INSTANCE;
            } else {
                return context.serialize(src.getUuid(), UUID.class);
            }
        }
    };
}
