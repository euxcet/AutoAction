package com.example.datacollection.utils;

import android.content.Context;
import android.util.Log;

import com.lzy.okgo.OkGo;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String UPLOAD_URL = "";
    private static final ThreadPoolExecutor executor = new ThreadPoolExecutor(
            1,
            2,
            10,
            TimeUnit.MINUTES,
            new LinkedBlockingQueue<Runnable>(),
            Executors.defaultThreadFactory(),
            new ThreadPoolExecutor.AbortPolicy()
            );

    public static void uploadFile(Context context, File file) {
        executor.execute(() -> OkGo.<String>post(UPLOAD_URL)
                .tag(context)
                .params("file", file)
                .isMultipart(true)
                .execute(new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) {
                        Log.e(TAG, file.getName() + " Upload Succeeded.");
                    }

                    @Override
                    public void onError(Response<String> response) {
                        Log.e(TAG, file.getName() + " Upload failed.");
                    }
                }));
    }
}
