package com.hcifuture.contextactionlibrary.contextaction.action;

import static java.lang.StrictMath.abs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.util.Log;

import com.hcifuture.contextactionlibrary.data.ProximitySensorManager;
import com.hcifuture.contextactionlibrary.model.NcnnInstance;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.util.List;

public class CloseAction extends BaseAction {

    //判断自然动作所需的变量
    boolean register_flag;
    float gx,gy,gz; //陀螺仪在x,y,z上的
    float dist; //接近光的数
    long register_time; //看接近光传感器注册的时间
    boolean upright_gyro; // 看凑近嘴部的加速度
    long up_gyro_id; // 凑近嘴部的gyro的id
    long success_id; //成功时的id
    boolean success_flag;

    public CloseAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener) {
        super(context, config, requestListener, actionListener);
    }

    //对变量进行初始化
    public void reset(){
        register_flag = false;
        register_time = -1;
        dist = -100;
        upright_gyro = false;
        up_gyro_id = -1;
        success_id = -1;
        success_flag = false;
    }

    @Override
    public void start() {
        isStarted = true;
        reset();
    }

    @Override
    public void stop() {
        isStarted = false;
    }

    @Override
    public void onIMUSensorChanged(SensorEvent event) {
        int type = event.sensor.getType();
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
                //需要将弧度转为角度
                gx = (float)Math.toDegrees(event.values[0]);
                gy = (float)Math.toDegrees(event.values[1]);
                gz = (float)Math.toDegrees(event.values[2]);
                // check_close
                //gx要必须很大，且gy，gz不大，才可以说明是凑近嘴部。
                if(!upright_gyro) {
//                Log.i("proximity:","gx: "+gx+" gy: "+gy+" gz: "+gz);
                    if (gx > 30 && abs(gx) - abs(gy) > 30 && abs(gx) - abs(gz) > 30) {
                        up_gyro_id = System.currentTimeMillis();
                        upright_gyro = true;
                        if (!register_flag) {
                            register_time = System.currentTimeMillis();
                            Log.i("proximity:", "注册传感器!" + register_time);
                            //TODO: 怎么注册传感器（和取消注册）
//                            ProximitySensorManager.start();
//                            sm.registerListener(this, sm.getDefaultSensor(Sensor.TYPE_PROXIMITY), SensorManager.SENSOR_DELAY_FASTEST);
                            register_flag = true;
                        }
                        Log.i("proximity:", "gyro为true了！");
                    }
                }
        }
    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {
        int type = event.sensor.getType();
        switch (type) {
            case Sensor.TYPE_PROXIMITY:
                dist = event.values[0];
                Log.i("proximity:","接近光距离为："+dist);
                if(dist == 0 ) {
                    if (upright_gyro) {
                        success_id = System.currentTimeMillis();
                        Log.i("proximity:", "识别成功2----"+(success_id-register_time)+" "+success_id);
                    } else {
                        Log.i("proximity:", "gyro不满足条件");
                    }
                }
                if(dist==5) {
                    success_id = -1;
                    if (System.currentTimeMillis() - register_time > 10000) {
                        //TODO: 取消传感器注册
//                    sm.unregisterListener(this, sm.getDefaultSensor(Sensor.TYPE_PROXIMITY));
                        Log.i("proximity:", "接近光 时间过长，取消注册传感器");
                        reset();
                        register_flag = false;
                    }
                }
        }

        if(success_id != -1){
            if(System.currentTimeMillis()-success_id>100){
                success_flag = true;
            }
            else{
                Log.i("proximity","稳定的时间不够"+(System.currentTimeMillis()-success_id));
            }
        }

    }

    @Override
    public void getAction() {
        if (!isStarted)
            return;
        if (NcnnInstance.getInstance() != null) {
            if (success_flag) {
                reset();
                for (ActionListener listener: actionListener) {
                    listener.onAction(new ActionResult("Close"));
                }
            }
        }
    }
}
