package com.hcifuture.contextactionlibrary.utils;

import android.content.Context;
import android.net.Uri;
import android.webkit.MimeTypeMap;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.google.gson.Gson;

import java.io.File;

import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String ROOT_URL = BuildConfig.WEB_SERVER;
    private static final String COLLECTED_DATA_URL = ROOT_URL + "/collected_data";

    private static Gson gson = new Gson();
    private static OkHttpClient client = new OkHttpClient();

    /*
        fileType:
            - 0 sensor bin
     */
    public static void uploadCollectedData(Context context, File file, int fileType, String name, String userId, long timestamp, String commit, Callback callback) {
        String extension = MimeTypeMap.getFileExtensionFromUrl(Uri.fromFile(file).toString());
        String mime = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension.toLowerCase());
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", file.getName(), RequestBody.create(MediaType.parse(mime), file))
                .addFormDataPart("fileType", String.valueOf(fileType))
                .addFormDataPart("userId", userId)
                .addFormDataPart("name", name)
                .addFormDataPart("timestamp", String.valueOf(timestamp))
                .addFormDataPart("commit", commit)
                .build();
        Request request = new Request.Builder()
                .url(COLLECTED_DATA_URL)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .post(requestBody)
                .build();
        client.newCall(request).enqueue(callback);

        /*
        OkGo.<String>post(COLLECTED_DATA_URL)
                .tag(context)
                .params("file", file)
                .params("fileType", fileType)
                .params("userId", userId)
                .params("name", name)
                .params("timestamp", timestamp)
                .params("commit", commit)
                .execute(callback);
         */
    }
}
