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

public class SubtaskAdapter extends BaseAdapter {
    private Context context;
    private LayoutInflater inflater;
    private TaskList task;
    private int task_id;

    public SubtaskAdapter(Context context, TaskList task, int task_id) {
        this.context = context;
        this.task = task;
        this.task_id = task_id;
        this.inflater = LayoutInflater.from(context);
    }

    @Override
    public int getCount() {
        return task.getTask().get(task_id).getSubtask().size();
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
        if (view == null) {
            view = inflater.inflate(R.layout.fragment_task, null);
            TaskList.Task.Subtask subtask = task.getTask().get(task_id).getSubtask().get(i);
            TextView textView = view.findViewById(R.id.textView);
            textView.setText(subtask.getName());
            /*
            view.setOnClickListener((v) -> {
                Bundle bundle = new Bundle();
                bundle.putInt("task_id", i);
                Intent intent = new Intent();
                intent.putExtras(bundle);
                context.startActivity(intent);
            });
             */
        }
        return view;
    }
}
