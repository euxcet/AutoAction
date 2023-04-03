package com.hcifuture.contextactionlibrary.contextaction.action;

import static android.os.VibrationEffect.createOneShot;
import static java.lang.StrictMath.abs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
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

import java.util.Deque;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorStatusHolder.CollectorStatus;

public class CloseAction extends BaseAction {
    public static String ACTION = "action.close.action";
    public static String ACTION_START = "action.close.start";

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

    float last_proxi;
    int x_stop_flag;
    private Vibrator vibrator;
    Deque<Float> gx_before = new LinkedList<>();

    CollectorStatus proximity_flag,light_flag;
    boolean send_flag; //假如没有传感器时，是否发送了日志
    String TAG = "proximity";


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
        last_proxi = -1;
        x_stop_flag = -1;
        gx_before.clear();
    }

    //识别成功时调用此函数
    private void Success(){
        //识别成功时，注册listener
        success_flag = true;
//        vibrate();
        //Log.e(TAG, "---------凑近嘴部------------");
   
    }

    private float[] movingaverage(int window_size){
        float[] data = new float[50-window_size+1];
        //对gx_before进行滑动平均
        float sum = 0; //所有的总和
        //滑动平均时的头和尾巴
        Iterator<Float> head = gx_before.iterator();
        Iterator<Float> tail = gx_before.iterator();
        int cnt = 0; //计数现在到第几个了
//        //Log.e("moving:", String.valueOf(gx_before.size()));
        while(tail.hasNext()){
            cnt ++;
            sum += tail.next();
//            //Log.e("moving:",cnt+" "+sum);
            if(cnt==window_size){
                data[cnt-window_size] = sum;
            }
            else if(cnt>window_size){
                sum -= head.next();
                data[cnt-window_size] = sum;
            }
            else{
                continue;
            }
//            //Log.e("moving:",cnt+" "+sum);
        }
        return data;
    }

    //查看gx是否是连续下降的
    private int[] get_num(){
        int num = 0;
        float tolerance = 1;
        int negative_num = 0;
        float[] data =movingaverage(20);
        ////Log.e("moving:", String.valueOf(data));
        for(int i=0;i<data.length;i++){
            if(data[i]<0){
                negative_num ++;
            }
            if(i==data.length-1){
                break;
            }
            if(data[i]>=data[i+1]){
                num ++;
            }
            else{
                if(data[i]>data[i+1]-5){
                    num +=0;
                }
                else if(data[i]>data[i+1]-10){
                    if(tolerance>0)
                        tolerance --;
                    else
                        num = 1;
                }
                else{
                    ////Log.e("close","gx突变太剧烈了"+data[i]+" "+data[i+1]);
                    //新增的
                    num = 1;
                }
            }
        }
        ////Log.e("moving:",num+" "+negative_num);
        return new int[]{num,negative_num};
    }

    private void vibrate(){
        VibrationEffect vibe = null;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            vibe = createOneShot(100,10);
            vibrator.vibrate(vibe);
        }
    }


    @Override
    public synchronized void start() {
        isStarted = true;
        proximity_flag = CollectorStatusHolder.getInstance().getStatus(Sensor.TYPE_PROXIMITY);
        ////Log.e(TAG,"proximity_flag:"+proximity_flag);
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
                gx = (float)Math.toDegrees(data.getValues().get(0));
                gy = (float)Math.toDegrees(data.getValues().get(1));
                gz = (float)Math.toDegrees(data.getValues().get(2));
                //50个
                if(gx_before.size()>=50){
                    gx_before.removeFirst();
                }
                gx_before.add(gx);
                int gx_gy_threshold =25;
                //gx要必须很大，且gy，gz不大，才可以说明是凑近嘴部。
                if(!upright_gyro) {
//                    //Log.e(TAG,s);

                    if (gx>50&&abs(gx)-abs(gy)>gx_gy_threshold &&abs(gx)-abs(gz)>30) {
                        up_gyro_id = System.currentTimeMillis(); // TODO：改成sensorEvent的时间
                        upright_gyro = true;
                        if(!register_flag){
                            register_time = System.currentTimeMillis();
//                            //Log.e(TAG,"注册传感器!"+register_time);
//                            sm.registerListener(this, sm.getDefaultSensor(Sensor.TYPE_PROXIMITY), SensorManager.SENSOR_DELAY_FASTEST);
                            register_flag = true;
                        }
                        //Log.e(TAG,"gyro为true了！");
                    }
                    else{
                        if(gx>50){
                            boolean b2 =  abs(gx)-abs(gy)> gx_gy_threshold?true:false;
                            boolean b3 =  abs(gx)-abs(gz)>30?true:false;
                            String s = "";
                            if(!b2){
                                s+=" b2: false,"+ (abs(gx)-abs(gy));
                            }
                            if(!b3){
                                s+=" b3: false,"+ (abs(gx)-abs(gz));
                            }
                            //Log.e(TAG,"gx满足，但其他不满足 "+s);
                        }
                    }
                }
                else if(upright_gyro){
//                    //Log.e(TAG,"gx:"+gx+" gy:"+gy+" gz:"+gz);
                    int gy_gx_threshold = 20;
                    int gy_threshold = 100;
                    if((abs(abs(gy)-abs(gx))<gy_gx_threshold||abs(abs(gz)-abs(gx))<30||(abs(gz)>abs(gx))||abs(gy)>abs(gx))&&(abs(gy)>gy_threshold||abs(gz)>60)){
                        upright_gyro = false;
//                        boolean b1 =  abs(abs(gy)-abs(gx))<gy_gx_threshold?true:false;
//                        boolean b2 =  abs(abs(gz)-abs(gx))<30?true:false;
//                        boolean b3 =  (abs(gz)>abs(gx))?true:false;
//                        boolean b4 = abs(gy)>abs(gx)?true:false;
//                        boolean b5 = abs(gy)>gy_threshold?true:false;
//                        boolean b6 = abs(gz)>60?true:false;
//                        String s = "";
//                        if(b1){
//                            s+=" b1:"+ abs(abs(gy)-abs(gx));
//                        }
//                        if(b2){
//                            s+=" b2: "+ abs(abs(gz)-abs(gx));
//                        }
//                        if(b3){
//                            s+=" b3: "+ (abs(gz)-abs(gx));
//                        }
//                        if(b4){
//                            s+=" b4: "+ (abs(gy)-abs(gx));
//                        }
//                        if(b5){
//                            s+=" b5:,"+ (abs(gy));
//                        }
//                        if(b6){
//                            s+=" b6:,"+ (abs(gz));
//                        }
                        //Log.e(TAG,"其他方向的角速度太大了："+s);
                    }
                    if(System.currentTimeMillis()-up_gyro_id>4000){
                        upright_gyro = false;
                        //Log.e(TAG,"角速度时间过长");
                    }
                }
                break;
            default:
                break;
        }
        check();
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {
        if(!send_flag) {
            if (logCollector != null) {
                logCollector.addLog("Proximity_sensor:" + proximity_flag);
                logCollector.addLog("Light_sensor:" + light_flag);
                for (ActionListener listener : actionListener) {
                    listener.onAction(new ActionResult(ACTION_START));
                }
                send_flag = true;
            }
        }
        if(proximity_flag != CollectorStatus.READY){
            //Log.e(TAG,"proximity not ready");
            return;
        }
        //在Log中添加凑近的接近光和光线传感器的数据
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

        Heart.getInstance().newActionAliveEvent(getConfig().getAction(), data.getTimestamp());

        int type = data.getType();
        switch (type) {
            case Sensor.TYPE_PROXIMITY:
                dist = data.getProximity();
                ////Log.e(TAG,"接近光距离为："+dist);
                if(dist == 0 ) { //凑近了
                    if (upright_gyro && last_proxi!=0) {
//                if (upright_gyro && last_proxi!=0 && gx<60 && gx>-25 && abs(gz)<25 && abs(gy)<55) {
                        int[] num = get_num();
                        if(num[0]<15 || num[1]>5){
                            //不满足条件
                            //Log.e("moving:","数量不满足条件"+num[0]+" "+num[1]);
                            reset();
                        }
                        else{
                            success_id = System.currentTimeMillis();
                            //Log.e(TAG, "识别成功2----"+(success_id-register_time)+" "+success_id);
                        }
                    } else {
                        if(last_proxi==5) { //稳定之后就不做这个判断了
                            boolean b1 = gx > 50 ? true : false;
                            boolean b2 = abs(gx) - abs(gy) > 30 ? true : false;
                            boolean b3 = abs(gx) - abs(gz) > 30 ? true : false;
                            String s = "";
                            if (!b1) {
                                s += " b1: false," + gx;
                            }
                            if (!b2) {
                                s += " b2: false," + (abs(gx) - abs(gy));
                            }
                            if (!b3) {
                                s += " b3: false," + (abs(gx) - abs(gz));
                            }
                            //Log.e(TAG, "gyro不满足条件" + s + " " + last_proxi);
                        }
                    }
                }
                if(dist==5) {
                    success_id = -1;
                    if (System.currentTimeMillis() - register_time > 10000 && register_flag) {
//                        sm.unregisterListener(this, sm.getDefaultSensor(Sensor.TYPE_PROXIMITY));
                        //Log.e(TAG, "接近光 时间过长，取消注册传感器");
                        reset();
                        register_flag = false;
                    }
                }
                last_proxi = dist;
                break;
            case Sensor.TYPE_LIGHT:
                if(dist == 5){ //距离远 但光传感器很小
                    if(data.getEnvironmentBrightness()<5){
                        //Log.e(TAG,"光传感器很小，重置");
                        reset();
                    }
                }
                break;
            default:
                break;
        }
        check();

    }

    public void check(){
        if(success_id != -1){
            ////Log.e(TAG,"stop:"+x_stop_flag+"gx:"+gx+" diff:"+(System.currentTimeMillis() - success_id));
            if(x_stop_flag == -1)
                if(System.currentTimeMillis() - success_id<100) {
                    if (gx < 50 && gx > -20 && abs(gz) < 20 && abs(gy) < 50) {
                        ////Log.e(TAG, "time:" + (System.currentTimeMillis() - success_id));
                        x_stop_flag = 1;
                        //Log.e(TAG,"已停下");
                    }
                }
                else
                    reset(); //太久都没有停止
            if(x_stop_flag==1) { //停止之后不能有大的变动
                if (gx > 50 || gx<-30 || abs(gy)>50 || abs(gz)>22) { //角速度太大
                    //Log.e(TAG,"停下后 角速度太大："+gx+" "+gy+" "+gz);
                    x_stop_flag = 0;
                    reset();
                }
            }

            if (System.currentTimeMillis() - success_id > 50) {//300) {
                if(x_stop_flag==1) {
                    if(gx<50 && gx>-20 && abs(gz)<10 ) {
//                        //Log.e(TAG, "成功了！" + success_id);
                        Success();
                    }
                }
            }
            else{
                ////Log.e(TAG,"稳定的时间不够"+(System.currentTimeMillis()-success_id));
            }
        }
    }

    @Override
    public void onExternalEvent(Bundle bundle) {

    }

    @Override
    public synchronized void getAction() {
        if (!isStarted)
            return;
        if (success_flag) {
            reset();
           // //Log.e(TAG,"识别成功了");
            for (ActionListener listener: actionListener) {
                listener.onAction(new ActionResult(ACTION));
            }
        }
    }

    @Override
    public String getName() {
        return "CloseAction";
    }
}
