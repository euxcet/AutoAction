package com.example.datacollection.ui.adapter;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.ui.ConfigSubtaskActivity;

public class TaskAdapter extends BaseAdapter {
    private Context context;
    private TaskList task;
    private LayoutInflater inflater;

    public TaskAdapter(Context context, TaskList task) {
        this.context = context;
        this.task = task;
        this.inflater = LayoutInflater.from(context);
    }

    @Override
    public int getCount() {
        return task.getTask().size();
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
        TextView taskName = view.findViewById(R.id.taskName);
        TextView taskTimes = view.findViewById(R.id.taskTimes);
        TextView taskDuration = view.findViewById(R.id.taskDuration);
        TextView taskVideo = view.findViewById(R.id.taskVideo);
        TextView taskAudio = view.findViewById(R.id.taskAudio);

        taskName.setText(task.getTask().get(i).getName());
        taskTimes.setText("  录制次数:     " + task.getTask().get(i).getTimes());
        taskDuration.setText("  单次时长:     " + task.getTask().get(i).getDuration() + " ms");
        taskVideo.setText("  开启摄像头: " + task.getTask().get(i).isVideo());
        taskAudio.setText("  开启麦克风: " + task.getTask().get(i).isAudio());

        view.setOnClickListener((v) -> {
            Bundle bundle = new Bundle();
            bundle.putInt("task_id", i);
            Intent intent = new Intent(context, ConfigSubtaskActivity.class);
            intent.putExtras(bundle);
            context.startActivity(intent);
        });
        return view;
    }
}
