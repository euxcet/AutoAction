package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.ui.adapter.TaskAdapter;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class ConfigTaskActivity extends AppCompatActivity {
    private ListView taskListView;

    private TaskListBean taskList;
    private TaskAdapter taskAdapter;

    private Context mContext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config);

        mContext = this;

        Button backButton = findViewById(R.id.taskBackButton);
        backButton.setOnClickListener((v) -> this.finish());

        Button addTaskButton = findViewById(R.id.addTaskButton);
        addTaskButton.setOnClickListener((v) -> {
            Intent intent = new Intent(ConfigTaskActivity.this, AddTaskActivity.class);
            startActivity(intent);
        });

        taskListView = findViewById(R.id.trainListView);
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadTaskListViaNetwork();
    }

    private void loadTaskListViaNetwork() {
        NetworkUtils.getTaskList(mContext, GlobalVariable.getInstance().getString("taskListId"), 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                taskList = new Gson().fromJson(response.body(), TaskListBean.class);
                taskAdapter = new TaskAdapter(mContext, taskList);
                taskListView.setAdapter(taskAdapter);
            }
        });
    }
}