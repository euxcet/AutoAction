package com.example.datacollection.visual;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;

import com.example.datacollection.R;
import com.example.datacollection.ui.adapter.TaskAdapter;
import com.example.datacollection.utils.GlobalVariable;
import com.example.datacollection.utils.NetworkUtils;
import com.example.datacollection.utils.RandomUtils;
import com.example.datacollection.utils.bean.RecordListBean;
import com.example.datacollection.utils.bean.StringListBean;
import com.example.datacollection.utils.bean.TaskListBean;
import com.example.datacollection.visual.adapter.RecordAdapter;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class RecordListActivity extends AppCompatActivity {
    private Context mContext;

    private RecordListBean recordList;
    private RecordAdapter recordAdapter;
    private ListView recordListView;
    private TaskListBean taskList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record_list);
        this.mContext = this;
        this.recordListView = findViewById(R.id.recordListView);
        Button backButton = findViewById(R.id.backButton);
        backButton.setOnClickListener((v) -> this.finish());
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadRecordListViaNetwork();
    }

    private void loadRecordListViaNetwork() {
        String taskListId = GlobalVariable.getInstance().getString("taskListId");
        NetworkUtils.getTaskList(mContext, taskListId, 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                taskList = new Gson().fromJson(response.body(), TaskListBean.class);
                NetworkUtils.getRecordList(mContext, taskListId, "0", "0", new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) {
                        recordList = new Gson().fromJson(response.body(), RecordListBean.class);
                        recordAdapter = new RecordAdapter(mContext, taskList, recordList);
                        recordListView.setAdapter(recordAdapter);
                    }
                });
            }
        });
    }
}