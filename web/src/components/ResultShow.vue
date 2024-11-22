<script setup lang="ts">
import * as echarts from 'echarts'
import type { PredictionResult } from '~/types/result'

const props = defineProps<{ result: PredictionResult[] | PredictionResult }>()

const illness = [
  'BipolarDisorder',
  'Schizophrenia',
  'Depression',
  'AnxietyDisorder',
  'PTSD',
]

const chartContainerRef = ref<HTMLDivElement | null>(null)
const predictionChartRef = ref<echarts.ECharts | null>(null)
type EChartsOption = echarts.EChartsOption

const option = computed<EChartsOption>(() => {
  return {
    color: Array.isArray(props.result) ? ['#FF917C', '#67F9D8', '#FFE434', '#56A3F1'] : ['#FF917C'],
    legend: {
      orient: 'horizontal', // 图例方向，可选 'horizontal' 或 'vertical'
      top: 'top', // 图例位置，例如 'top', 'bottom', 'left', 'right' 或具体像素值
      left: '200', // 图例居中
    },
    radar: [
      {
        indicator: illness.map((item) => {
          return {
            text: item,
            max: 100,
          }
        }),
        radius: '60%',
        center: ['55%', '60%'],
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
        tooltip: {
          trigger: 'item',
        },
        emphasis: {
          lineStyle: {
            width: 4,
          },
          label: {
            show: true,
            position: [0, 0],
            fontSize: 20,
          },
          focus: 'series',
        },
        data: Array.isArray(props.result)
          ? props.result.map((item) => {
            return {
              value: Object.values(item.result),
              name: item.name,
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
            }
          })
          : [{
              value: Object.values(props.result.result),
              name: props.result.name,
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
            }],
        areaStyle: {
          opacity: 0.1,
        },
      },
    ],
  }
})
function resizeChart(chart: echarts.ECharts) {
  chart.resize()
}

watch(option, (newValue) => {
  predictionChartRef.value?.setOption(newValue, true)
})
onMounted(() => {
  const chartDom = chartContainerRef.value
  if (chartDom && option){
    predictionChartRef.value = echarts.init(chartDom)
    predictionChartRef.value.setOption(option.value)
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
  <div ref="chartContainerRef" class="chartContainer" />
</template>

<style scoped>
.chartContainer {
  position: relative;
  height: 100%;
  width: 100%;
  overflow: hidden;
}
</style>
