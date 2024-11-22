<script setup lang="ts">
import { CheckOutlined, CloseOutlined } from '@ant-design/icons-vue'
import { usePredictionResultStore } from '~/store/predictionResult'
import { modelEval } from '~/types/model'

const integrateRadar = ref(false)
const justShowCNN = ref(false)
const predictionStore = usePredictionResultStore()
</script>

<template>
  <div class="final">
    <div class="switchBox">
      <a-switch v-model:checked="integrateRadar">
        <template #checkedChildren>
          <div class="switchInner">
            <p>Integrate Radar</p>
            <CheckOutlined />
          </div>
        </template>
        <template #unCheckedChildren>
          <div class="switchInner">
            <p>Segregate Radar</p>
            <CloseOutlined />
          </div>
        </template>
      </a-switch>
      <a-switch v-if="integrateRadar" v-model:checked="justShowCNN">
        <template #checkedChildren>
          <div class="switchInner">
            <p>Show Your Risk</p>
            <CheckOutlined />
          </div>
        </template>
        <template #unCheckedChildren>
          <div class="switchInner">
            <p>Show Your Risk</p>
            <CloseOutlined />
          </div>
        </template>
      </a-switch>
    </div>
    <div class="resultBox">
      <div v-if="integrateRadar" class="radarIntegrateContainer">
        <ResultShow :result="justShowCNN ? predictionStore.results[0] : predictionStore.results" />
      </div>
      <div v-for="(result) in predictionStore.results" v-else :key="result.key" class="radarSegContainer">
        <ResultShow :result="result" />
      </div>
    </div>
    <div class="evalBox">
      <EvalShow :data="modelEval.map((item) => { return item.F1Score })" />
      <EvalShow :data="modelEval.map((item) => { return item.Accuracy })" />
    </div>
  </div>
</template>

<style scoped lang="scss">
.final {
  height: 100%;
  display: flex;
  position: relative;
  .switchBox {
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-size: 13px;
    top: 5px;
    left: 10px;
    position: absolute;
    z-index: 100;
  }
}
.chartContainer {
  position: relative;
  height: 100%;
  width: 100%;
  overflow: hidden;
}
.resultBox {
  height: 100%;
  width: 60%;
  position: relative;
  display: flex;
  flex-wrap: wrap;
}
.evalBox {
  height: 100%;
  width: 40%;
}
.radarSegContainer {
  width: 50%;
  height: 50%;
}
.radarIntegrateContainer {
  width: 100%;
  height: 100%;
}
.switchInner {
  display: flex;
  align-items: center;
  gap: 5px;
  p {
    margin: 0;
  }
}
</style>
