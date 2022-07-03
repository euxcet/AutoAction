package com.hcifuture.datacollection.data;

import android.content.Context;

import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * The controller for managing timestamp data.
 */
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

    /**
     * Cancel the ongoing subtask as if it has never been started.
     */
    public void cancel() {
        data.clear();
    }

    public void stop() {
        FileUtils.writeStringToFile(new Gson().toJson(data), this.saveFile);
    }

    public void add(long timestamp) {
        data.add(timestamp);
    }

    public void upload(String taskListId, String taskId, String subtaskId, String recordId, long timestamp) {
        if (saveFile != null) {
            NetworkUtils.uploadRecordFile(mContext, saveFile, TaskListBean.FILE_TYPE.TIMESTAMP.ordinal(), taskListId, taskId, subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }
    }
}
