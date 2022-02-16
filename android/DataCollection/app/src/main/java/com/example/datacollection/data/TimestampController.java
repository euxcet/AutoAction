package com.example.datacollection.data;

import android.content.Context;

import com.example.datacollection.utils.FileUtils;
import com.example.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TimestampController {
    private Context mContext;
    private List<Long> data = new ArrayList<>();
    private File saveFile;

    public TimestampController(Context context) {
        this.mContext = context;
    }

    public void start(File file) {
        this.saveFile = file;
        data.clear();
    }

    public void stop() {
        FileUtils.writeStringToFile(new Gson().toJson(data), this.saveFile);
    }

    public void add(long timestamp) {
        data.add(timestamp);
    }

    public void upload() {
        if (saveFile != null) {
            NetworkUtils.uploadFile(mContext, saveFile);
        }
    }
}
