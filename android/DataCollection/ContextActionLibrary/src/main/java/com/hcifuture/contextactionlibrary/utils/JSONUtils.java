package com.hcifuture.contextactionlibrary.utils;

import android.os.Bundle;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.shared.communicate.result.Result;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class JSONUtils {

    public static void jsonPut(JSONObject json, String key, Object value) {
        try {
            json.put(key, value);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    public static Object wrapObjRecursive(Object obj) {
        if (obj == null) {
            return null;
        } else if (obj.getClass().isArray() &&
                obj.getClass().getComponentType() != null &&
                !obj.getClass().getComponentType().isPrimitive()) {
            // exclude primitive type array
            JSONArray array = new JSONArray();
            for (Object tmp : (Object[]) obj) {
                array.put(wrapObjRecursive(tmp));
            }
            return array;
        } else {
            Object obj_wrap = JSONObject.wrap(obj);
            if (obj_wrap == null) {
                obj_wrap = JSONObject.wrap(obj.toString());
            }
            return obj_wrap;
        }
    }

    public static void jsonPutBundle(JSONObject json, Bundle bundle) {
        if (bundle != null && json != null) {
            for (String key : bundle.keySet()) {
                Object obj = bundle.get(key);
                if (obj instanceof Bundle) {
                    jsonPut(json, key, bundleToJSON((Bundle) obj));
                } else {
                    jsonPut(json, key, wrapObjRecursive(obj));
                }
            }
        }
    }

    public static JSONObject bundleToJSON(Bundle bundle) {
        if (bundle == null) {
            return null;
        } else {
            JSONObject json = new JSONObject();
            jsonPutBundle(json, bundle);
            return json;
        }
    }

    public static void mapPutBundle(Map<String, Object> map, Bundle bundle) {
        if (bundle != null && map != null) {
            for (String key : bundle.keySet()) {
                Object obj = bundle.get(key);
                if (obj instanceof Bundle) {
                    map.put(key, bundleToMap((Bundle) obj));
                } else {
                    map.put(key, obj);
                }
            }
        }
    }

    public static Map<String, Object> bundleToMap(Bundle bundle) {
        if (bundle == null) {
            return null;
        } else {
            Map<String, Object> map = new HashMap<>();
            mapPutBundle(map, bundle);
            return map;
        }
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
            Bundle bundle = result.getExtras();
            mapPutBundle(map, bundle);
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
            mapPutBundle(map, bundle);
            return map;
        }
    }
}
