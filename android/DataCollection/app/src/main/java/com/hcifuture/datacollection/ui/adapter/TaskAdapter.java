package com.hcifuture.datacollection.ui.adapter;

import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.ui.NormalAlertDialog;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.ui.ConfigSubtaskActivity;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class TaskAdapter extends BaseAdapter {
    private Context mContext;
    private TaskListBean taskList;
    private LayoutInflater inflater;

    public TaskAdapter(Context context, TaskListBean taskList) {
        this.mContext = context;
        this.taskList = taskList;
        this.inflater = LayoutInflater.from(context);
    }

    @Override
    public int getCount() {
        return taskList.getTask().size();
    }

    @Override
    public Object getItem(int i) {
        return null;
    }

    @Override
    public long getItemId(int i) {
        return 0;
    }

    @Override
    public View getView(int i, View view, ViewGroup viewGroup) {
        view = inflater.inflate(R.layout.fragment_task, null);
        TextView taskId = view.findViewById(R.id.taskId);
        TextView taskName = view.findViewById(R.id.taskName);
        TextView taskTimes = view.findViewById(R.id.taskTimes);
        TextView taskDuration = view.findViewById(R.id.taskDuration);
        TextView taskVideo = view.findViewById(R.id.taskVideo);
        TextView taskAudio = view.findViewById(R.id.taskAudio);
        Button deleteButton = view.findViewById(R.id.deleteItemButton);

        TaskListBean.Task task = taskList.getTask().get(i);
        taskName.setText(task.getName());
        taskId.setText("  编号:            " + task.getId());
        taskTimes.setText("  录制次数:     " + task.getTimes());
        taskDuration.setText("  单次时长:     " + task.getDuration() + " ms");
        taskVideo.setText("  开启摄像头: " + task.isVideo());
        taskAudio.setText("  开启麦克风: " + task.isAudio());

        deleteButton.setOnClickListener((v) -> {
            NormalAlertDialog dialog = new NormalAlertDialog(mContext,
                    "Delete task [" + task.getId() + "] ?",
                    "");
            dialog.setPositiveButton("Yes",
                    (dialogInterface, i1) -> {
                        String id = task.getId();
                        for(int j = 0; j < taskList.getTask().size(); j++) {
                            if (taskList.getTask().get(j).getId().equals(id)) {
                                taskList.getTask().remove(j);
                            }
                        }
                        NetworkUtils.updateTaskList(mContext, taskList, 0, new StringCallback() {
                            @Override
                            public void onSuccess(Response<String> response) {
                            }
                        });
                        this.notifyDataSetChanged();
                    });
            dialog.setNegativeButton("No",
                    (dialogInterface, i12) -> dialog.dismiss());
            dialog.create();
            dialog.show();
        });

        view.setOnClickListener((v) -> {
            Bundle bundle = new Bundle();
            bundle.putInt("task_id", i);
            Intent intent = new Intent(mContext, ConfigSubtaskActivity.class);
            intent.putExtras(bundle);
            mContext.startActivity(intent);
        });
        return view;
    }
}
