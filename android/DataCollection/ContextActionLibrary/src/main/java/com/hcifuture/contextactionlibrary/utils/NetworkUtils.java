package com.hcifuture.contextactionlibrary.utils;

import android.content.Context;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.google.gson.Gson;
import com.lzy.okgo.OkGo;
import com.lzy.okgo.callback.StringCallback;

import java.io.File;

public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String ROOT_URL = BuildConfig.WEB_SERVER;
    private static final String COLLECTED_DATA_URL = ROOT_URL + "/collected_data";

    private static Gson gson = new Gson();

    /*
        fileType:
            - 0 sensor bin
     */
    public static void uploadCollectedData(Context context, File file, int fileType, String name, String userId, long timestamp, String commit, StringCallback callback) {
        OkGo.<String>post(COLLECTED_DATA_URL)
                .tag(context)
                .params("file", file)
                .params("fileType", fileType)
                .params("userId", userId)
                .params("name", name)
                .params("timestamp", timestamp)
                .params("commit", commit)
                .execute(callback);
    }
}
