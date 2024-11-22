<script lang="ts" setup>
import HeaderView from './layout/HeaderView.vue'
import { requestSymptom } from './requestWrapper'
import { useSymptomStore } from './store/symptom'

const symptomStore = useSymptomStore()

onMounted(async () => {
  const symptomRaw = await requestSymptom()
  symptomStore.setSymptom(symptomRaw)
  symptomStore.initNewHistory()
})
</script>

<template>
  <div class="layout">
    <HeaderView />
    <div class="main">
      <RouterView />
    </div>
  </div>
</template>

<style lang="less" scoped>
.layout {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.main {
  flex: 1;
  height: 0;
}
</style>
