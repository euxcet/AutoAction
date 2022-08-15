package com.hcifuture.datacollection.ui.register;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.action.ActionManager;

public class RegisterActivity extends AppCompatActivity {
    private TextView actionListView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);
        setRoute(R.id.captureButton, RegisterCaptureActivity.class);
        setRoute(R.id.knockButton, RegisterKnockActivity.class);
        setRoute(R.id.pointButton, RegisterPointActivity.class);
        setRoute(R.id.placeButton, RegisterPlaceActivity.class);
        setRoute(R.id.leanButton, RegisterLeanActivity.class);
        setRoute(R.id.autoButton, RegisterAutoActivity.class);
        actionListView = findViewById(R.id.actionListView);
        refreshActionList();
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshActionList();
    }

    private void setRoute(int buttonId, Class activityClass) {
        Button button = findViewById(buttonId);
        button.setOnClickListener((v) -> {
            Intent intent = new Intent(RegisterActivity.this, activityClass);
            startActivity(intent);
        });
    }

    private void refreshActionList() {
        actionListView.setText(ActionManager.encodeActions(
                ActionManager.getInstance().getActions()));
    }

}