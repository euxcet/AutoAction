package com.hcifuture.datacollection.ui.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.bean.TrainListBean;


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

        return view;
    }
}
