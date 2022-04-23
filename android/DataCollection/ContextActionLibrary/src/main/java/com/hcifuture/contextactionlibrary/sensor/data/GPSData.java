package com.hcifuture.contextactionlibrary.sensor.data;

import java.util.ArrayList;
import java.util.List;

public class GPSData extends Data {
    private List<SatelliteData> satellites;
    private double latitude;
    private double longitude;
    private double altitude;
    private float accuracy;
    private float bearing;
    private float speed;
    private float time;
    private String provider;
    private int satelliteCount;

    public GPSData deepClone() {
        GPSData result = new GPSData();
        result.setLatitude(latitude);
        result.setLongitude(longitude);
        result.setAltitude(altitude);
        result.setAccuracy(accuracy);
        result.setBearing(bearing);
        result.setSpeed(speed);
        result.setTime(time);
        result.setProvider(provider);
        result.setSatelliteCount(satelliteCount);
        if (satellites != null) {
            List<SatelliteData> satelliteData = new ArrayList<>();
            for (SatelliteData satellite: satellites) {
                satelliteData.add(satellite.deepClone());
            }
            result.setSatellites(satelliteData);
        }
        return result;
    }

    public int getSatelliteCount() {
        return satelliteCount;
    }

    public void setSatelliteCount(int satelliteCount) {
        this.satelliteCount = satelliteCount;
    }

    public double getAltitude() {
        return altitude;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    public float getAccuracy() {
        return accuracy;
    }

    public float getBearing() {
        return bearing;
    }

    public float getSpeed() {
        return speed;
    }

    public float getTime() {
        return time;
    }

    public String getProvider() {
        return provider;
    }

    public void setAccuracy(float accuracy) {
        this.accuracy = accuracy;
    }

    public void setAltitude(double altitude) {
        this.altitude = altitude;
    }

    public void setBearing(float bearing) {
        this.bearing = bearing;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    public void setProvider(String provider) {
        this.provider = provider;
    }

    public void setSpeed(float speed) {
        this.speed = speed;
    }

    public void setTime(float time) {
        this.time = time;
    }

    @Override
    public DataType dataType() {
        return DataType.GPSData;
    }

    public List<SatelliteData> getSatellites() {
        return satellites;
    }

    public void setSatellites(List<SatelliteData> satellites) {
        this.satellites = satellites;
    }

    public static class SatelliteData {
        private int constellationType;
        private int svid;
        private float azimuthDegrees;
        private float carrierFrequencyHz;
        private float cn0DbHz;
        private float elevationDegrees;

        public SatelliteData(int constellationType, int svid, float azimuthDegrees, float carrierFrequencyHz, float cn0DbHz, float elevationDegrees) {
            this.constellationType = constellationType;
            this.svid = svid;
            this.azimuthDegrees = azimuthDegrees;
            this.carrierFrequencyHz = carrierFrequencyHz;
            this.cn0DbHz = cn0DbHz;
            this.elevationDegrees = elevationDegrees;
        }

        public SatelliteData deepClone() {
            return new SatelliteData(constellationType, svid, azimuthDegrees, carrierFrequencyHz, cn0DbHz, elevationDegrees);
        }

        public float getCn0DbHz() {
            return cn0DbHz;
        }

        public void setCn0DbHz(float cn0DbHz) {
            this.cn0DbHz = cn0DbHz;
        }

        public float getAzimuthDegrees() {
            return azimuthDegrees;
        }

        public float getCarrierFrequencyHz() {
            return carrierFrequencyHz;
        }

        public float getElevationDegrees() {
            return elevationDegrees;
        }

        public int getConstellationType() {
            return constellationType;
        }

        public int getSvid() {
            return svid;
        }

        public void setAzimuthDegrees(float azimuthDegrees) {
            this.azimuthDegrees = azimuthDegrees;
        }

        public void setCarrierFrequencyHz(float carrierFrequencyHz) {
            this.carrierFrequencyHz = carrierFrequencyHz;
        }

        public void setConstellationType(int constellationType) {
            this.constellationType = constellationType;
        }

        public void setElevationDegrees(float elevationDegrees) {
            this.elevationDegrees = elevationDegrees;
        }

        public void setSvid(int svid) {
            this.svid = svid;
        }
    }
}
