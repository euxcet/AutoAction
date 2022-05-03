package com.hcifuture.contextactionlibrary.utils;

import android.os.Bundle;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.shared.communicate.result.Result;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class JSONUtils {

    public static void silentPut(JSONObject json, String key, Object value) {
        try {
            json.put(key, value);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    public static void silentPutBundle(JSONObject json, Bundle bundle) {
        for (String key : bundle.keySet()) {
            Object obj = JSONObject.wrap(bundle.get(key));
            if (obj == null) {
                obj = JSONObject.wrap(bundle.get(key).toString());
            }
            silentPut(json, key, obj);
        }
    }

    public static JSONObject bundleToJSON(Bundle bundle) {
        JSONObject json = new JSONObject();
        silentPutBundle(json, bundle);
        return json;
    }

    public static Map<String, Object> collectorResultToMap(CollectorResult result) {
        if (result == null)
            return null;
        else {
            Map<String, Object> map = new HashMap<>();
            map.put("StartTimestamp", result.getStartTimestamp());
            map.put("EndTimestamp", result.getEndTimestamp());
            map.put("Type", result.getType());
            map.put("ErrorCode", result.getErrorCode());
            map.put("ErrorReason", result.getErrorReason());
            return map;
        }
    }

    public static Map<String, Object> resultToMap(Result result) {
        if (result == null)
            return null;
        else {
            Map<String, Object> map = new HashMap<>();
            map.put("Key", result.getKey());
            map.put("Timestamp", result.getTimestamp());
            map.put("Reason", result.getReason());
            Bundle bundle = result.getExtras();
            for (String key : bundle.keySet()) {
                map.put(key, bundle.get(key));
            }
            return map;
        }
    }
}
