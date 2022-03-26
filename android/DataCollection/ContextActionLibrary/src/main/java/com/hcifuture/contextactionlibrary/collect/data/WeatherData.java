package com.hcifuture.contextactionlibrary.collect.data;

public class WeatherData extends Data {
    String reportTime;
    String weather;
    String windDirection;
    String windPower;
    float temperature;
    float humidity;

    public WeatherData() {
        reportTime = "";
        weather = "";
        windDirection = "";
        windPower = "";
        temperature = 0;
        humidity = 0;
    }

    public WeatherData(String reportTime, String weather, String windDirection, String windPower,
                       float temperature, float humidity) {
        this.reportTime = reportTime;
        this.weather = weather;
        this.windDirection = windDirection;
        this.windPower = windPower;
        this.temperature = temperature;
        this.humidity = humidity;
    }

    public float getHumidity() {
        return humidity;
    }

    public float getTemperature() {
        return temperature;
    }

    public String getReportTime() {
        return reportTime;
    }

    public String getWeather() {
        return weather;
    }

    public String getWindDirection() {
        return windDirection;
    }

    public String getWindPower() {
        return windPower;
    }

    public void setHumidity(float humidity) {
        this.humidity = humidity;
    }

    public void setReportTime(String reportTime) {
        this.reportTime = reportTime;
    }

    public void setTemperature(float temperature) {
        this.temperature = temperature;
    }

    public void setWeather(String weather) {
        this.weather = weather;
    }

    public void setWindDirection(String windDirection) {
        this.windDirection = windDirection;
    }

    public void setWindPower(String windPower) {
        this.windPower = windPower;
    }
}
