package com.example.simpleexample.contextaction.utils;

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
    private static final String FILE_URL = ROOT_URL + "/file";
    private static final String MD5_URL = ROOT_URL + "/md5";

    private static Gson gson = new Gson();

    public static void downloadFile(Context context, String filename, FileCallback callback) {
        OkGo.<File>get(FILE_URL)
                .tag(context)
                .params("filename", filename)
                .execute(callback);
    }

    public static void getMD5(Context context, String filename, StringCallback callback) {
        OkGo.<String>get(MD5_URL)
                .tag(context)
                .params("filename", filename)
                .execute(callback);
    }
}
