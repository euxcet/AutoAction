package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;

import com.example.datacollection.R;
import com.example.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class TrainActivity extends AppCompatActivity {
    private static final String TAG = "TrainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_train);
        Button button = findViewById(R.id.newProgramButton);
        button.setOnClickListener(view -> {
            Intent intent = new Intent(TrainActivity.this, NewTrainingProgramActivity.class);
            startActivity(intent);
            /*
            NetworkUtils.getTaskList(this, "TL13r912je", 0, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                    Log.e(TAG, response.body());
                }
            });
             */
        });
    }
}