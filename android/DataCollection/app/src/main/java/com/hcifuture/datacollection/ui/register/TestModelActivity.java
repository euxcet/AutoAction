package com.hcifuture.datacollection.ui.register;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.util.Pair;
import android.widget.Button;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.action.ActionEnum;
import com.hcifuture.datacollection.action.ActionManager;
import com.hcifuture.datacollection.action.ActionResult;
import com.hcifuture.datacollection.action.ActionWithObject;
import com.hcifuture.datacollection.data.CameraController;
import com.hcifuture.datacollection.inference.ImuEventListener;
import com.hcifuture.datacollection.inference.ImuSensorManager;
import com.hcifuture.datacollection.inference.Inferencer;
import com.hcifuture.datacollection.inference.OrientationManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The activity to test a model.
 * Jumped from MainActivity.
 */
public class TestModelActivity extends AppCompatActivity implements ImuEventListener {

    private AppCompatActivity mActivity;
    private ImuSensorManager mImuSensorManager;
    private OrientationManager mOrientationManager;
    private CameraController mCameraController;
    private TextView actionListView;
    private TextView frameResultView;
    private Vibrator mVibrator;
    private List<ActionResult> resultHistory = new ArrayList<>();

    private AtomicBoolean knockAtomic = new AtomicBoolean(false);
    private ActionEnum lastActionEnum;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test_model);
        mImuSensorManager = new ImuSensorManager(this);
        mImuSensorManager.addListener(this);
        mOrientationManager = new OrientationManager(this);
        mActivity = this;

        mCameraController = new CameraController(mActivity);
        mCameraController.openCamera(1, false);

        mVibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        TextView modelIdView = findViewById(R.id.actionListView);
        modelIdView.setText("Model Id: " + Inferencer.getInstance().getCurrentModelId());
        TextView labelView = findViewById(R.id.labelTextView);
        labelView.setText("Label: " + Inferencer.getInstance().getActionLabels());

        Button registerButton = findViewById(R.id.registerButton);
        registerButton.setOnClickListener((v) -> {
            Intent intent = new Intent(TestModelActivity.this, RegisterActivity.class);
            startActivity(intent);
        });

        actionListView = findViewById(R.id.actionListView);
        frameResultView = findViewById(R.id.frameResultView);

//        new Thread(() -> detectFrame()).start();
    }

    private void detectFrame() {
        mCameraController.capture().whenComplete((v, e) -> {
            if (knockAtomic.get()) {
                knockAtomic.set(false);
                ActionResult result = ActionManager.getInstance().classify(v.getFeature(), lastActionEnum);
                if (result.getAction() != null) {
                    addResult(result);
                }
            }
//            detectFrame();
        });
    }

    private void addResult(ActionResult result) {
        resultHistory.add(result);
        if (resultHistory.size() > 10) {
            resultHistory.remove(0);
        }
        StringBuilder str = new StringBuilder("Result:\n");
        for (ActionResult r: resultHistory) {
            str.append(r.getTimestamp())
                    .append(" ")
                    .append(r.getAction().getName())
                    .append(" ")
                    .append(r.getAction().getAction())
                    .append(" ")
                    .append(r.getMinDistance())
                    .append("\n");
        }
        runOnUiThread(() -> frameResultView.setText(str.toString()));
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mImuSensorManager != null) {
            mImuSensorManager.stop();
        }
        if (mOrientationManager != null) {
            mOrientationManager.stop();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshActionList();
        if (mImuSensorManager != null) {
            mImuSensorManager.start();
        }
        if (mOrientationManager != null) {
            mOrientationManager.start();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mImuSensorManager != null) {
            mImuSensorManager.stop();
        }
        if (mOrientationManager != null) {
            mOrientationManager.stop();
        }
    }

    private void refreshActionList() {
        actionListView.setText(ActionManager.encodeActions(
                ActionManager.getInstance().getActions()));
    }

    @Override
    public void onStatus(String status) {

    }

    @Override
    public void onAction(String action) {
        ActionEnum actionEnum = ActionEnum.Place;
        switch (action) {
            case "point":
                actionEnum = ActionEnum.Point;
                break;
            case "capture":
                actionEnum = ActionEnum.Capture;
                break;
            case "lean":
                actionEnum = ActionEnum.Lean;
                break;
            case "knock":
                actionEnum = ActionEnum.Knock;
                break;
            case "point_in":
                actionEnum = ActionEnum.PointIn;
                break;
            case "point_out":
                actionEnum = ActionEnum.PointOut;
                break;
        }
        addResult(new ActionResult(new ActionWithObject("Default", actionEnum, null), 0, 0));
        ActionEnum finalActionEnum = actionEnum;
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                if (!knockAtomic.get()) {
                    lastActionEnum = finalActionEnum;
                    knockAtomic.set(true);
                    detectFrame();
                }
            }
        }, 300);
        mVibrator.vibrate(VibrationEffect.createOneShot(200, 128));
    }
}