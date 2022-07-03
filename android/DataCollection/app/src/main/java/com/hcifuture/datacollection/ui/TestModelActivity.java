package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.inference.IMUSensorManager;
import com.hcifuture.datacollection.inference.Inferencer;

/**
 * The activity to test a model.
 * Jumped from MainActivity.
 */
public class TestModelActivity extends AppCompatActivity {

    private IMUSensorManager imuSensorManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test_model);
        imuSensorManager = new IMUSensorManager(this);
        imuSensorManager.start();
        TextView modelIdView = findViewById(R.id.modelIdTextView);
        modelIdView.setText("Model Id: " + Inferencer.getInstance().getCurrentModelId());
        TextView labelView = findViewById(R.id.labelTextView);
        labelView.setText("Label: " + Inferencer.getInstance().getLabel());
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (imuSensorManager != null) {
            imuSensorManager.stop();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (imuSensorManager != null) {
            imuSensorManager.start();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (imuSensorManager != null) {
            imuSensorManager.stop();
        }
    }
}