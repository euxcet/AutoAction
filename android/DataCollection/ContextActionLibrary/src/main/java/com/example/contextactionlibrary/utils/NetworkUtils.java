package com.example.contextactionlibrary.utils;

import android.content.Context;

import com.google.gson.Gson;
import com.lzy.okgo.OkGo;
import com.lzy.okgo.callback.StringCallback;

import java.io.File;

public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String ROOT_URL = "http://192.168.31.186:60010";
    private static final String COLLECTED_DATA_URL = ROOT_URL + "/collected_data";

    private static Gson gson = new Gson();

    public static void uploadCollectedData(Context context, File file, int fileType, String name, String commit, StringCallback callback) {
        OkGo.<String>post(COLLECTED_DATA_URL)
                .tag(context)
                .params("file", file)
                .params("fileType", fileType)
                .params("name", name)
                .params("commit", commit)
                .execute(callback);
    }
}
