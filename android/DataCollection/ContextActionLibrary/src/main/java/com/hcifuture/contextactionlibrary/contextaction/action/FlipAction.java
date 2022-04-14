package com.hcifuture.contextactionlibrary.contextaction.action;


import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.StrictMath.abs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.util.Log;

import com.hcifuture.contextactionlibrary.model.NcnnInstance;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.util.List;

public class FlipAction extends BaseAction{

    //判断自然动作所需的变量
    boolean flag1,flag2; //翻转的第1步和第2步的角度判断
    boolean gyro_flag1,gyro_flag2; //角速度对flip进行判断
    long flag1_id,gyro_id1,gyro_id2,flag2_id; //两次稳态的时间
    double pitch;
    float gx,gy,gz; //陀螺仪在x,y,z上的
    boolean postive; //第一次是否为正数
    float max_gy,max_gx; //最大的gy的值，避免出现gx过大的情况
    long gx_id; //最大gx的下标
    private float[] values, r, gravity, geomagnetic;
    boolean success_flag;

    public FlipAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener) {
        super(context, config, requestListener, actionListener);
    }

    //对变量进行初始化
    private void reset(){
        values = new float[3];//用来保存最终的结果
        gravity = new float[3];//用来保存加速度传感器的值
        r = new float[9];//
        geomagnetic = new float[3];//用来保存地磁传感器的值
        flag1 = false; flag2 = false;
        gyro_flag1 = false; gyro_flag2 = false;
        gyro_id1 = -1; gyro_id2 = -1;
        flag1_id = -1; flag2_id = -1;
        max_gy = 0; max_gx = 0;
        gx_id = -1;
        success_flag = false;
    }

    private boolean check_gy(){
        if(max_gx>max_gy+100 && gx_id>min(gyro_id1,flag1_id) && gx_id<max(gyro_id2,flag2_id)){
            Log.i("FLIP","gy不满足条件");
            return false;
        }
        Log.i("FLIP","gy满足条件");
        return true;
    }

    @Override
    public synchronized void start() {
        isStarted = true;
        reset();
    }

    @Override
    public synchronized void stop() {
        isStarted = false;
    }

    @Override
    public synchronized void onIMUSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            gravity = event.values.clone();
            getValue(); //更新方位角
            if(abs(gravity[2])>20){
                Log.i("FLIP","角速度过大"+abs(gravity[2]));
                reset();
            }
        }
        else if(event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD){
            geomagnetic = event.values.clone();
            getValue();
        }
        else if(event.sensor.getType() == Sensor.TYPE_GYROSCOPE){
            //需要将弧度转为角度
            gx = (float)Math.toDegrees(event.values[0]);
            gy = (float)Math.toDegrees(event.values[1]);
            gz = (float)Math.toDegrees(event.values[2]);
            if(abs(gy)>max_gy+100){
                max_gx = abs(gx);
                gx_id = System.currentTimeMillis();
            }

            if(abs(gy)>160){
                max_gy = max(max_gy,abs(gy));
                if(!gyro_flag1){
                    postive = gy>0?true:false;
                    gyro_flag1 = true;
                    gyro_id1 = System.currentTimeMillis();
                    Log.i("FLIP","角速度1:第一次大于200:"+"gy: "+gy+" gx: "+gx+"gz:" +gz+"ID："+gyro_id1);
                }
                else if(!gyro_flag2){
                    Log.i("FLIP","角速度2:第二次大于200"+"gy: "+gy+" gx: "+gx+"gz:" +gz+"ID："+gyro_id1);
                    if(postive){
                        if(gy<0 && flag1){
                            if(System.currentTimeMillis()-gyro_id1<20000) {
                                gyro_flag2 = true;
                                gyro_id2 = System.currentTimeMillis();
                                Log.i("FLIP","角速度2:第二次为正"+gyro_id2);
                                Log.i("FLIP","-------------------角速度ok");
                                if(flag2 && (gyro_id2-flag2_id)<100){
                                    Log.i("FLIP","角速度3:角度也满足条件！");
                                    if(check_gy()) {
                                        success_flag=true;
//                                        Success();
                                    }
                                }
                            }
                            else{
                                Log.i("FLIP","角速度4:距离上次200的时间太长");
                                gyro_id1 = System.currentTimeMillis();
                                postive = false;
                            }
                        }
                        else{
                            Log.i("FLIP","角速度5:更新上一次为正");
                            gyro_id1 = System.currentTimeMillis();
                            postive = true;
                        }
                    }
                    else{
                        if(gy>0){
                            if(System.currentTimeMillis()-gyro_id1<20000) {
                                gyro_flag2 = true;
                                gyro_id2 = System.currentTimeMillis();
                                Log.i("FLIP","角速度2:第二次为负");
                                Log.i("FLIP","-------------------角速度ok");
                                if(flag2 && (gyro_id2-flag2_id)<3000){
                                    Log.i("FLIP","角速度3:角度也满足条件！");
                                    if(check_gy())
                                        success_flag = true;
//                                        Success();
                                }
                            }
                            else{
                                Log.i("FLIP","角速度4:距离上次200的时间太长");
                                gyro_id1 = System.currentTimeMillis();
                                postive = true;
                            }
                        }
                        else{
                            Log.i("FLIP","角速度5:更新上一次为负");
                            gyro_id1 = System.currentTimeMillis();
                            postive = false;
                        }
                    }
                }
                else{
                    if(((postive && gy<0)||(!postive && gy>0))&&(System.currentTimeMillis()-gyro_id1<30000)){
                        gyro_id2 = System.currentTimeMillis();
                        Log.i("FLIP","角速度7：更新gyro2"+gy+"id2:"+gyro_id2);
                    }
                    else {
                        //假如gyro_flag1和2 都满足了，就要考虑更新gyro_flag1和2
                        //可能是用户第一次没识别出来，要通过第二次来识别，假如这时不更新，只能等到时长太长再更新了！
                        postive = gy > 0 ? true : false;
                        gyro_flag1 = true;
                        gyro_id1 = System.currentTimeMillis();
                        Log.i("FLIP", "角速度6:第三次大于200"+"postive:"+postive+"gy:"+gy);
                        gyro_flag2 = false;
                    }
                }
            }
            if(gyro_flag1&&gyro_flag2){
                if(System.currentTimeMillis()-gyro_id2>10000){
                    Log.i("FLIP","角速度6:时间太长了 角速度的两个flag更新为false");
                    gyro_flag1 = false;gyro_flag2 = false;
                }
            }

        }


    }

    //获取方位角数据
    private void getValue() {
        // r从这里返回
//        Log.i("FLIP","更新方向角");
        if(gravity!=null && geomagnetic!=null){
            if(SensorManager.getRotationMatrix(r, null, gravity, geomagnetic)) {
                //values从这里返回
                SensorManager.getOrientation(r, values);
                //提取数据
                double azimuth = Math.toDegrees(values[0]);
                if (azimuth < 0) {
                    azimuth = azimuth + 360;
                }
                pitch = Math.toDegrees(values[1]);
                double roll = Math.toDegrees(values[2]);
//                Log.i("FLIP","roll: "+Math.floor(roll)+"ptich: "+Math.floor(pitch));

                if(pitch>40){
                    reset();
                    Log.i("FLIP:","pitch太大了，重置"+pitch);
                }

                if(!flag1) {
                    if(gyro_flag1) {
                        if (pitch > -60 && pitch < 35) { //保证是平面 , <35是因为当速度很快时 就会让pitch很大
                            //TODO: 快的时候，很多次 roll不在阈值范围里面
//                        Log.i("FLIP","角度0:满足pitch条件" + roll);
                            if ((roll < -100 && roll > -180) || (roll > 100 && roll < 180)) { //保证翻转了180度
                                flag1 = true;
                                flag1_id = System.currentTimeMillis();
                                Log.i("FLIP", "角度1:翻转到下面了"+(flag1_id-gyro_id1)+" "+flag1_id+" "+gyro_id1);
                            }
//                        else{
//                            Log.i("FLIP","roll的角度不满足:"+roll);
//                        }
                        }
                    }
                }
                //flag
                else{
//                    if(!(pitch>-65 && pitch<35)) {
//                        flag1 = false;
//                        Log.i("FLIP","角度2:太竖直了 失败:"+pitch);
////                        flip_cnt = 0;
//                    }
                    if(roll<30 && roll>-30 && pitch > -60 && pitch < 15){
                        flag1 = false;
                        Log.i("FLIP","角度3:返回平面了");
                        if(System.currentTimeMillis() - flag1_id<20000) {
                            Log.i("FLIP","角度4:时间满足");
                            //检查gyro满足条件没 , 只要gyro_id2满足条件就行， 其实不用检查gyro_flag2
                            if((System.currentTimeMillis()-gyro_id2)<10000) {
                                Log.i("FLIP","角度5:角速度也满足条件啦！");
                                if(check_gy())
                                    success_flag = true;
//                                    Success();
                            }
                            else{
                                Log.i("FLIP","角度6:角速度未满足条件"+(System.currentTimeMillis()-gyro_id2)+"f1:"+gyro_flag1+"f2:"+gyro_flag2);
                                Log.i("FLIP","-------------------角度ok");
                                flag2 = true;
                                flag2_id = System.currentTimeMillis();
                            }
                        }
                    }
//                    else{
//                        Log.i("FLIP","roll: "+Math.floor(roll)+"ptich: "+Math.floor(pitch));
//                    }
                }
            }
        }
    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {
    }

    @Override
    public synchronized void getAction() {
        if (!isStarted)
            return;
        if (success_flag) {
            Log.i("FLIP","识别成功了");
            reset();
            for (ActionListener listener: actionListener) {
                listener.onAction(new ActionResult("Flip"));
            }
        }
    }

}
