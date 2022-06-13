package com.hcifuture.contextactionlibrary.contextaction.action;

import static java.lang.StrictMath.abs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorStatusHolder;
import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorStatusHolder.CollectorStatus;

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
    float bright;
    CollectorStatus proximity_flag,light_flag;
    boolean send_flag; //假如没有传感器时，是否发送了日志

    public CloseAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, LogCollector CloseLogCollector) {
        super(context, config, requestListener, actionListener, scheduledExecutorService, futureList);
        logCollector = CloseLogCollector;
        reset();
    }

    //对变量进行初始化
    private void reset(){
        register_flag = false;
        register_time = -1;
        dist = -100;
        upright_gyro = false;
        up_gyro_id = -1;
        success_id = -1;
        success_flag = false;
        bright = -1;
    }

    @Override
    public synchronized void start() {
        isStarted = true;
        proximity_flag = CollectorStatusHolder.getInstance().getStatus(Sensor.TYPE_PROXIMITY);
        light_flag = CollectorStatusHolder.getInstance().getStatus(Sensor.TYPE_LIGHT);
        send_flag = false;
        reset();

    }

    @Override
    public synchronized void stop() {
        isStarted = false;
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
        if(proximity_flag != CollectorStatus.READY){
            return;
        }
        Heart.getInstance().newActionAliveEvent(getConfig().getAction(), data.getTimestamp());
        int type = data.getType();
        switch (type) {
            case Sensor.TYPE_GYROSCOPE:
                // 需要将弧度转为角度
                gx = (float)Math.toDegrees(data.getValues().get(0));
                gy = (float)Math.toDegrees(data.getValues().get(1));
                gz = (float)Math.toDegrees(data.getValues().get(2));
                // check_close
                // gx要必须很大，且gy，gz不大，才可以说明是凑近嘴部。
                if (!upright_gyro) {
//                Log.i("proximity:","gx: "+gx+" gy: "+gy+" gz: "+gz);
                    if (gx > 30 && abs(gx) - abs(gy) > 30 && abs(gx) - abs(gz) > 30) {
                        up_gyro_id = System.currentTimeMillis();
                        upright_gyro = true;
                        if (!register_flag) {
                            register_time = System.currentTimeMillis();
                            Log.i("proximity:", "注册传感器!" + register_time);
                            //TODO: 怎么注册传感器（和取消注册）
//                            for (ActionListener listener: actionListener) {
//                                listener.onAction(new ActionResult("START_PROXIMITY"));
//                            }
//                            sm.registerListener(this, sm.getDefaultSensor(Sensor.TYPE_PROXIMITY), SensorManager.SENSOR_DELAY_FASTEST);
                            register_flag = true;
                        }
                        Log.i("proximity:", "gyro为true了！");
                    }
                }
                if(upright_gyro){
//                    Log.i("proximity:","gx:"+gx+" gy:"+gy+" gz:"+gz);
                    if((abs(abs(gy)-abs(gx))<30||abs(abs(gz)-abs(gx))<30||(abs(gz)>abs(gx))||abs(gy)>abs(gx))&&(abs(gy)>60||abs(gz)>60)){
                        upright_gyro = false;
                        Log.i("proximity:","其他方向的角速度太大了");
                    }
                    if(System.currentTimeMillis()-up_gyro_id>4000){
                        upright_gyro = false;
                        Log.i("proximity:","角速度时间过长");
                    }
                }
        }

        if (success_id != -1) {
            if(!success_flag) {
                if (System.currentTimeMillis() - success_id > 100) {
                    Log.i("proximity:", "成功了！" + success_id);
                    success_flag = true;
                }
            }
//            else {
//                Log.i("proximity","稳定的时间不够" + (System.currentTimeMillis() - success_id));
//            }
        }
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {
        if(!send_flag) {
            if (logCollector != null) {
                logCollector.addLog("Proximity_sensor:" + proximity_flag);
                logCollector.addLog("Light_sensor:" + light_flag);
                for (ActionListener listener : actionListener) {
                    listener.onAction(new ActionResult("CloseStart"));
                }
                send_flag = true;
            }
        }
        if(proximity_flag != CollectorStatus.READY){
            return;
        }

        Heart.getInstance().newActionAliveEvent(getConfig().getAction(), data.getTimestamp());

        if(data.getType()==Sensor.TYPE_PROXIMITY){
            dist = data.getProximity();
        }
        if(data.getType()==Sensor.TYPE_LIGHT){
            bright = data.getEnvironmentBrightness();
        }
        if(upright_gyro) {
            if (logCollector != null) {
                if(data.getType() == Sensor.TYPE_PROXIMITY)
                    logCollector.addLog(data.getProximityTimestamp() + " " + dist + " " + bright);
                else if(data.getType() == Sensor.TYPE_LIGHT) {
                    logCollector.addLog(data.getEnvironmentBrightnessTimestamp() + " " + dist + " " + bright);
                }
            }
        }
        if(dist == 0 ) {
            if (upright_gyro) {
                success_id = System.currentTimeMillis();
                Log.i("proximity:", "识别成功2----"+(success_id-register_time)+" "+success_id);
            }
//            else {
//                Log.i("proximity:", "gyro不满足条件");
//            }
        }
        if(dist==5) {
            success_id = -1;
            if (System.currentTimeMillis() - register_time > 10000 && register_flag) {
                //TODO: 取消传感器注册
//                for (ActionListener listener : actionListener) {
//                    listener.onAction(new ActionResult("STOP_PROXIMITY"));
//                }
//                    sm.unregisterListener(this, sm.getDefaultSensor(Sensor.TYPE_PROXIMITY));
                Log.i("proximity:", "接近光 时间过长，取消注册传感器");
                reset();
                register_flag = false;
            }
        }
    }

    @Override
    public synchronized void getAction() {
        if (!isStarted)
            return;
        if (success_flag) {
            reset();
            Log.i("proximity:","识别成功了");
            for (ActionListener listener: actionListener) {
                listener.onAction(new ActionResult("Close"));
            }
        }
    }
}
