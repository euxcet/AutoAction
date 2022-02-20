package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ListView;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.ui.adapter.TaskAdapter;

public class ConfigTaskActivity extends AppCompatActivity {
    private ListView taskListView;
    private Button backButton;
    private Button addTaskButton;

    private TaskList taskList;
    private TaskAdapter taskAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config);

        backButton = findViewById(R.id.taskBackButton);
        backButton.setOnClickListener((v) -> this.finish());

        addTaskButton = findViewById(R.id.addTaskButton);
        addTaskButton.setOnClickListener((v) -> {
            Intent intent = new Intent(ConfigTaskActivity.this, AddTaskActivity.class);
            startActivity(intent);
        });

        taskListView = findViewById(R.id.taskListView);

        taskList = TaskList.parseFromLocalFile();
        taskAdapter = new TaskAdapter(this, taskList);
        taskListView.setAdapter(taskAdapter);
    }

    @Override
    protected void onResume() {
        super.onResume();
        taskList = TaskList.parseFromLocalFile();
        taskAdapter = new TaskAdapter(this, taskList);
        taskListView.setAdapter(taskAdapter);
    }
}