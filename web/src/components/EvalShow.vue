<script setup lang="ts">
import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'

const props = defineProps<{ data: number[] }>()
const chartContainerRef = ref<HTMLDivElement | null>(null)
const evalChartRef = ref<echarts.ECharts | null>(null)
const option: EChartsOption = {
  xAxis: {
    axisLabel: {
      rotate: 45,
    },
    type: 'category',
    data: [{
      value: 'CNN Hierarchy',
      textStyle: {
        fontSize: 12,
      },
    }, {
      value: 'CNN Random',
      textStyle: {
        fontSize: 12,
      },
    }, {
      value: 'FFNN',
      textStyle: {
        fontSize: 12,
      },
    }, {
      value: 'XGBoost',
      textStyle: {
        fontSize: 12,
      },
    }],
    axisTick: {
      alignWithLabel: true,
    },
  },
  yAxis: {
    type: 'value',
    min: 30,
  },
  tooltip: {
    show: true,
  },
  series: [
    {
      barWidth: 20,
      data: props.data,
      type: 'bar',
      label: {
        show: true, // 显示标签
        position: 'inside', // 标签位置，可选 'top', 'inside', 'left', 等
        formatter: '{c}', // {c} 表示当前数据值
      },
      markPoint: {
        data: [
          { type: 'max', name: '最大值' },
        ],
      },
      markLine: {
        data: [
          { type: 'average', name: '平均值' },
        ],
      },
    },
  ],
}
function resizeChart(chart: echarts.ECharts) {
  chart.resize()
}
onMounted(() => {
  const chartContainerDom = chartContainerRef.value
  if (chartContainerDom && option){
    evalChartRef.value = echarts.init(chartContainerDom)
    evalChartRef.value.setOption(option)
    window.addEventListener('resize', () => {
      resizeChart(evalChartRef.value as echarts.ECharts)
    })
  }
})
onUnmounted(() => {
  window.removeEventListener('resize', () => {
    resizeChart(evalChartRef.value as echarts.ECharts)
  })
  evalChartRef.value?.dispose()
})
</script>

<template>
  <div ref="chartContainerRef" class="evalChart" />
</template>

<style scoped>
.evalChart {
  height: 50%;
}
</style>
