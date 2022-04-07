package com.hcifuture.datacollection.ui.adapter;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.ui.ConfigSubtaskActivity;
import com.hcifuture.datacollection.ui.NormalAlertDialog;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.TrainListBean;
import com.hcifuture.datacollection.visual.VisualizeTrainLogActivity;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;


public class TrainAdapter extends BaseAdapter {
    private Context mContext;
    private TrainListBean trainList;
    private LayoutInflater inflater;

    public TrainAdapter(Context context, TrainListBean trainList) {
        this.mContext = context;
        this.trainList = trainList;
        this.inflater = LayoutInflater.from(context);
    }

    @Override
    public int getCount() {
        return trainList.getTrainList().size();
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
        view = inflater.inflate(R.layout.fragment_train, null);
        TextView trainId = view.findViewById(R.id.trainId);
        TextView trainName = view.findViewById(R.id.trainName);
        TextView trainStatus = view.findViewById(R.id.trainStatus);

        TrainListBean.TrainBean train = trainList.getTrainList().get(i);
        trainName.setText(train.getName());
        trainId.setText("  编号:            " + train.getId());
        trainStatus.setText("  状态:            " + train.getStatus());

        Button deleteButton = view.findViewById(R.id.interruptButton);
        deleteButton.setOnClickListener((v) -> {
            NormalAlertDialog dialog = new NormalAlertDialog(mContext,
                    "Interrupt training program [" + trainId + "] ?",
                    "");
            dialog.setPositiveButton("Yes",
                    (dialogInterface, i1) -> NetworkUtils.stopTrain(mContext, train.getId(), System.currentTimeMillis(), new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                        }
                    }));
            dialog.setNegativeButton("No",
                    (dialogInterface, i12) -> dialog.dismiss());
            dialog.create();
            dialog.show();
        });
        if (!train.getStatus().equals("Training")) {
            deleteButton.setVisibility(View.INVISIBLE);
        }

        view.setOnClickListener((v) -> {
            Bundle bundle = new Bundle();
            bundle.putString("trainId", train.getId());
            Intent intent = new Intent(mContext, VisualizeTrainLogActivity.class);
            intent.putExtras(bundle);
            mContext.startActivity(intent);
        });

        return view;
    }
}
