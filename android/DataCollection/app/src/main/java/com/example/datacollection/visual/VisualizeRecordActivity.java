package com.example.datacollection.visual;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.widget.Button;

import com.example.datacollection.R;

public class VisualizeRecordActivity extends AppCompatActivity {
    private Context mContext;

    private String taskListId;
    private String taskId;
    private String subtaskId;
    private String recordId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_visualize_record);

        this.mContext = this;

        Bundle bundle = getIntent().getExtras();
        taskListId = bundle.getString("taskListId");
        taskId = bundle.getString("taskId");
        subtaskId = bundle.getString("subtaskId");
        recordId = bundle.getString("recordId");
    }
}