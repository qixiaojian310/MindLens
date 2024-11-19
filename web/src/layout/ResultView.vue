<script setup lang="ts">
import * as echarts from 'echarts'

const chartCOntainerRef = ref<HTMLDivElement | null>(null)
const predictionChartRef = ref<echarts.ECharts | null>(null)
type EChartsOption = echarts.EChartsOption

const option: EChartsOption = {
  color: ['#FF917C'],
  title: {
    text: 'Customized Radar Chart',
  },
  legend: {},
  radar: [
    {
      indicator: [
        { text: 'Indicator1', max: 150 },
        { text: 'Indicator2', max: 150 },
        { text: 'Indicator3', max: 150 },
        { text: 'Indicator4', max: 120 },
        { text: 'Indicator5', max: 108 },
      ],
      center: ['75%', '50%'],
      axisName: {
        color: '#fff',
        backgroundColor: '#666',
        borderRadius: 3,
        padding: [3, 5],
      },
    },
  ],
  series: [
    {
      type: 'radar',
      radarIndex: 0,
      data: [
        {
          value: [120, 118, 130, 100, 99],
          name: 'Data C',
          symbol: 'rect',
          symbolSize: 12,
          lineStyle: {
            type: 'dashed',
          },
          label: {
            show: true,
            formatter(params: any) {
              return params.value as string
            },
          },
        },
      ],
      areaStyle: {
        opacity: 0.1,
      },
    },
  ],
}
function resizeChart(chart: echarts.ECharts) {
  chart.resize()
}

onMounted(() => {
  const chartDom = chartCOntainerRef.value
  if (chartDom && option){
    predictionChartRef.value = echarts.init(chartDom)
    predictionChartRef.value.setOption(option)
    window.addEventListener('resize', () => {
      resizeChart(predictionChartRef.value as echarts.ECharts)
    })
  }
})
onUnmounted(() => {
  window.removeEventListener('resize', () => {
    resizeChart(predictionChartRef.value as echarts.ECharts)
  })
  predictionChartRef.value?.dispose()
})
</script>

<template>
  <div ref="chartCOntainerRef" class="chartContainer" />
</template>

<style scoped>
.chartContainer {
  position: relative;
  height: 100%;
  width: 100%;
  overflow: hidden;
}
</style>
