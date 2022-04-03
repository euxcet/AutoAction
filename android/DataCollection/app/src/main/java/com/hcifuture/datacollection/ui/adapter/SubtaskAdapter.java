package com.hcifuture.datacollection.ui.adapter;

import android.content.Context;
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
import com.hcifuture.datacollection.ui.ModifySubtaskActivity;
import com.hcifuture.datacollection.ui.NormalAlertDialog;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.util.List;

public class SubtaskAdapter extends BaseAdapter {
    private Context mContext;
    private LayoutInflater inflater;
    private TaskListBean taskList;
    private int task_id;

    public SubtaskAdapter(Context context, TaskListBean taskList, int task_id) {
        this.mContext = context;
        this.taskList = taskList;
        this.task_id = task_id;
        this.inflater = LayoutInflater.from(context);
    }

    @Override
    public int getCount() {
        return taskList.getTask().get(task_id).getSubtask().size();
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

        List<TaskListBean.Task.Subtask> subtasks = taskList.getTask().get(task_id).getSubtask();
        TaskListBean.Task.Subtask subtask = taskList.getTask().get(task_id).getSubtask().get(i);

        TextView taskId = view.findViewById(R.id.taskId);
        TextView taskName = view.findViewById(R.id.taskName);
        TextView taskTimes = view.findViewById(R.id.taskTimes);
        TextView taskDuration = view.findViewById(R.id.taskDuration);
        TextView taskVideo = view.findViewById(R.id.taskVideo);
        TextView taskAudio = view.findViewById(R.id.taskAudio);

        taskName.setText(subtask.getName());
        taskId.setText("  编号:            " + subtask.getId());
        taskTimes.setText("  录制次数:     " + subtask.getTimes());
        taskDuration.setText("  单次时长:     " + subtask.getDuration() + " ms");
        taskVideo.setText("  开启摄像头: " + subtask.isVideo());
        taskAudio.setText("  开启麦克风: " + subtask.isAudio());

        Button deleteButton = view.findViewById(R.id.deleteItemButton);

        deleteButton.setOnClickListener((v) -> {
            NormalAlertDialog dialog = new NormalAlertDialog(mContext,
                    "Delete subtask [" + subtask.getId() + "] ?",
                    "");
            dialog.setPositiveButton("Yes",
                    (dialogInterface, i1) -> {
                        String id = subtask.getId();
                        for(int j = 0; j < subtasks.size(); j++) {
                            if (subtasks.get(j).getId() == id) {
                                subtasks.remove(j);
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
            bundle.putInt("task_id", task_id);
            bundle.putInt("subtask_id", i);
            Intent intent = new Intent(mContext, ModifySubtaskActivity.class);
            intent.putExtras(bundle);
            mContext.startActivity(intent);
        });

        return view;
    }
}
