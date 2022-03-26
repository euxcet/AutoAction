package com.example.simpleexample.utils;

import android.content.Context;

import com.example.simpleexample.BuildConfig;
import com.google.gson.Gson;
import com.lzy.okgo.OkGo;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.callback.StringCallback;

import java.io.File;

public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String ROOT_URL = BuildConfig.WEB_SERVER;
    private static final String DOWNLOAD_FILE_URL = ROOT_URL + "/download_file";

    private static Gson gson = new Gson();

    public static void downloadFile(Context context, String filename, FileCallback callback) {
        OkGo.<File>get(DOWNLOAD_FILE_URL)
                .tag(context)
                .params("filename", filename)
                .execute(callback);

    }
}
