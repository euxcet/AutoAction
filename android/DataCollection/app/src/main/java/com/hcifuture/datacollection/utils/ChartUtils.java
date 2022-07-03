package com.hcifuture.datacollection.utils;

import android.graphics.Color;
import android.graphics.DashPathEffect;

import com.hcifuture.datacollection.visual.DataSetConfig;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineDataSet;

import java.util.ArrayList;

/**
 * Show line chart for visualization.
 */
public class ChartUtils {
    public static LineDataSet createLineDataSet(LineChart chart, ArrayList<Entry> values, DataSetConfig config) {
        LineDataSet lineDataSet = new LineDataSet(values, config.label);

        lineDataSet.setDrawIcons(false);

        lineDataSet.enableDashedLine(10f, 5f, 0f);

        lineDataSet.setColor(config.color);
        lineDataSet.setCircleColor(config.color);

        lineDataSet.setLineWidth(1f);

        lineDataSet.setDrawCircles(false);
        lineDataSet.setDrawCircleHole(false);

        lineDataSet.setFormLineWidth(1f);
        lineDataSet.setFormLineDashEffect(new DashPathEffect(new float[]{10f, 5f}, 0f));
        lineDataSet.setFormSize(15.f);

        lineDataSet.setValueTextSize(9f);

        lineDataSet.enableDashedHighlightLine(10f, 5f, 0f);

        lineDataSet.setDrawFilled(false);
        lineDataSet.setFillFormatter((dataSet, dataProvider) -> chart.getAxisLeft().getAxisMinimum());

        lineDataSet.setFillColor(Color.BLACK);

        return lineDataSet;
    }
}
