package com.example.datacollection.ui.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;

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
        TextView textView = view.findViewById(R.id.textView);
        textView.setText(task.getTask().get(i).getName());
        return view;
    }
}
