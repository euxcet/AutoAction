package com.hcifuture.datacollection.visual.adapter;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.bean.RecordListBean;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.visual.VisualizeRecordActivity;

import java.text.SimpleDateFormat;
import java.util.Date;

public class RecordAdapter extends BaseAdapter {
    private Context mContext;
    private TaskListBean mTaskList;
    private RecordListBean mRecordList;
    private LayoutInflater mInflater;

    public RecordAdapter(Context context, TaskListBean taskList, RecordListBean recordList) {
        this.mContext = context;
        this.mTaskList = taskList;
        this.mRecordList = recordList;
        this.mInflater = LayoutInflater.from(context);
    }

    @Override
    public int getCount() {
        return mRecordList.getRecordList().size();
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
        view = mInflater.inflate(R.layout.fragment_record, null);
        TextView recordId = view.findViewById(R.id.recordId);
        TextView taskId = view.findViewById(R.id.taskId);
        TextView subtaskId = view.findViewById(R.id.subtaskId);
        TextView timestamp = view.findViewById(R.id.timestamp);

        RecordListBean.RecordBean record = mRecordList.getRecordList().get(i);
        String v_taskId = record.getTaskId();
        String v_subtaskId = record.getSubtaskId();

        recordId.setText(record.getRecordId());
        taskId.setText("Task Name: " + mTaskList.getTaskNameById(v_taskId));
        if (mTaskList.getTaskById(v_taskId) != null) {
            subtaskId.setText("SubTask Name: " + mTaskList.getTaskById(v_taskId).getSubtaskNameById(v_subtaskId));
        } else {
            subtaskId.setText("SubTask Name: null");
        }

        Date date = new Date(record.getTimestamp());
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        timestamp.setText("Timestamp: " + format.format(date));

        // click the record item to visualize data
        view.setOnClickListener((v) -> {
            Bundle bundle = new Bundle();
            bundle.putString("taskListId", record.getTaskListId());
            bundle.putString("recordId", record.getRecordId());
            bundle.putString("taskId", record.getTaskId());
            bundle.putString("subtaskId", record.getSubtaskId());
            Intent intent = new Intent(mContext, VisualizeRecordActivity.class);
            intent.putExtras(bundle);
            mContext.startActivity(intent);
        });
        return view;
    }
}
