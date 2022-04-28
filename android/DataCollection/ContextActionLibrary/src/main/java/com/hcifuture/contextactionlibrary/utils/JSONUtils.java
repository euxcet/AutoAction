package com.hcifuture.contextactionlibrary.utils;

import android.os.Bundle;

import org.json.JSONException;
import org.json.JSONObject;

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

    public static JSONObject bundle2JSON(Bundle bundle) {
        JSONObject json = new JSONObject();
        silentPutBundle(json, bundle);
        return json;
    }

}
