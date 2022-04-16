package com.hcifuture.contextactionlibrary.sensor.data;

public class LocationData extends Data {
    private double longitude;
    private double latitude;
    private double altitude;
    private float accuracy;
    private String floor;
    private String city;
    private String poiName;
    private String street;
    private long time;
    private String cityCode;
    private String adCode;

    public LocationData() {
        longitude = 0;
        latitude = 0;
        altitude = 0;
        accuracy = 0;
        floor = "";
        city = "";
        poiName = "";
        street = "";
        cityCode = "";
        adCode = "";
    }

    public LocationData(double longitude, double latitude, double altitude,
                        float accuracy,
                        String floor, String city, String poiName, String street, long time, String adCode, String cityCode) {
        this.longitude = longitude;
        this.latitude = latitude;
        this.altitude = altitude;
        this.accuracy = accuracy;
        this.floor = floor;
        this.city = city;
        this.poiName = poiName;
        this.street = street;
        this.time = time;
        this.adCode = adCode;
        this.cityCode = cityCode;
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

    public String getCity() {
        return city;
    }

    public String getFloor() {
        return floor;
    }

    public String getPoiName() {
        return poiName;
    }

    public String getStreet() {
        return street;
    }

    public void setAccuracy(float accuracy) {
        this.accuracy = accuracy;
    }

    public void setAltitude(double altitude) {
        this.altitude = altitude;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public void setFloor(String floor) {
        this.floor = floor;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    public void setPoiName(String poiName) {
        this.poiName = poiName;
    }

    public void setStreet(String street) {
        this.street = street;
    }

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
    }

    public String getCityCode() {
        return cityCode;
    }

    public LocationData setCityCode(String cityCode) {
        this.cityCode = cityCode;
        return this;
    }

    public String getAdCode() {
        return adCode;
    }

    public LocationData setAdCode(String adCode) {
        this.adCode = adCode;
        return this;
    }

    @Override
    public DataType dataType() {
        return DataType.LocationData;
    }
}
