package com.hcifuture.datacollection.ui.register;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.data.CameraController;
import com.hcifuture.datacollection.inference.ImuSensorManager;
import com.hcifuture.datacollection.inference.Inferencer;

import java.util.Timer;
import java.util.TimerTask;

/**
 * The activity to test a model.
 * Jumped from MainActivity.
 */
public class TestModelActivity extends AppCompatActivity {

    private AppCompatActivity mActivity;
    private ImuSensorManager mImuSensorManager;
    private CameraController mCameraController;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test_model);
//        mImuSensorManager = new ImuSensorManager(this);
        mActivity = this;

        mCameraController = new CameraController(mActivity);
        mCameraController.openCamera(1, false);

        TextView modelIdView = findViewById(R.id.modelIdTextView);
        modelIdView.setText("Model Id: " + Inferencer.getInstance().getCurrentModelId());
        TextView labelView = findViewById(R.id.labelTextView);
        labelView.setText("Label: " + Inferencer.getInstance().getLabel());

        Button registerButton = findViewById(R.id.registerButton);
        registerButton.setOnClickListener((v) -> {
            Intent intent = new Intent(TestModelActivity.this, RegisterActivity.class);
            startActivity(intent);
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mImuSensorManager != null) {
            mImuSensorManager.stop();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mImuSensorManager != null) {
            mImuSensorManager.start();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mImuSensorManager != null) {
            mImuSensorManager.stop();
        }
    }
}