<script setup lang="ts">
import { ArrowLeftOutlined, BarsOutlined, CheckOutlined, CloseOutlined, HistoryOutlined } from '@ant-design/icons-vue'
import { Avatar, BadgeRibbon, Button, Divider, Drawer, Input, Modal, Switch } from 'ant-design-vue'
import { v4 as uuidv4 } from 'uuid'
import LOGO from '~/components/LOGO.vue'
import { requestPrediction } from '~/requestWrapper'
import type { SymptomKey } from '~/requestWrapper/utils'
import { convertSymptom } from '~/requestWrapper/utils'
import { usePredictionResultStore } from '~/store/predictionResult'
import type { HistoriesItem, History } from '~/store/stateType'
import { useSymptomStore } from '~/store/symptom'

const router = useRouter()
const symptomStore = useSymptomStore()
const predictionStore = usePredictionResultStore()
const open = ref<boolean>(false)
const modalOpen = ref<boolean>(false)
const showHistoryDetail = ref<boolean>(false)
const historyLoaded = ref<HistoriesItem>()
const isNewHistory = ref<boolean>(false)

function afterOpenChange(bool: boolean) {
  console.log('open', bool)
}
function loadHistory(history: HistoriesItem, isNew: boolean){
  console.log(history)
  historyLoaded.value = history
  showHistoryDetail.value = true
  isNewHistory.value = isNew
}
function showHistoryAll(){
  showHistoryDetail.value = false
  historyLoaded.value = undefined
}
function showDrawer() {
  open.value = true
}
function backToHome() {
  router.push({ name: 'Home' })
}

const evaluationName = ref<string>('New evaluation 1')

async function handleOk(e: MouseEvent) {
  if (historyLoaded.value?.history){
    console.log(e)
    modalOpen.value = false
    const predictionResult = await requestPrediction({
      features: historyLoaded.value?.history,
      worker_choice: 'cnn',
    })
    symptomStore.addHistories(historyLoaded.value.history, evaluationName.value)
    predictionStore.setPredictionResults(predictionResult)
    symptomStore.initIndex()
    symptomStore.initNewHistory()
    showHistoryDetail.value = false
    router.push({ name: 'Result' })
  }
}

async function evalHistory(newHistory: History | undefined){
  if (newHistory){
    if (isNewHistory.value){
      modalOpen.value = true
    }
    else {
      const predictionResult = await requestPrediction({
        features: newHistory,
        worker_choice: 'cnn',
      })
      predictionStore.setPredictionResults(predictionResult)
      symptomStore.initIndex()
      router.push({ name: 'Result' })
    }
  }
}

const newTempHistory = computed(() => {
  return {
    history: symptomStore.newHistory,
    name: 'New Temp History',
    key: uuidv4(),
  }
})
</script>

<template>
  <div class="header">
    <div class="logo" @click="backToHome">
      <div class="icon">
        <LOGO :size="90" />
      </div>
      <p class="title">
        MindLens
      </p>
    </div>
    <div class="historyDrawer">
      <Button :icon="h(BarsOutlined)" @click="showDrawer" />
      <Drawer
        v-model:open="open"
        class="custom-class"
        root-class-name="root-class-name"
        placement="right"
        :width="showHistoryDetail ? 500 : 400"
        @after-open-change="afterOpenChange"
      >
        <template #title>
          <div class="drawerTitle">
            <div>
              <p>{{ historyLoaded ? historyLoaded?.name : 'Histories' }}</p>
            </div>
            <div class="switchContainer">
              <Button v-if="showHistoryDetail" :icon="h(ArrowLeftOutlined)" @click="showHistoryAll" />
            </div>
          </div>
        </template>
        <template v-if="!showHistoryDetail">
          <div v-for="history in symptomStore.histories" :key="history.key" @click="loadHistory(history, false)">
            <div class="historyItem">
              <Avatar :icon="h(HistoryOutlined)" shape="square" />
              <p>{{ history.name }}</p>
            </div>
          </div>
          <BadgeRibbon text="Unsubmitted" color="red">
            <div @click="loadHistory(newTempHistory, true)">
              <div class="historyItem">
                <Avatar :icon="h(HistoryOutlined)" shape="square" />
                <p>{{ newTempHistory.name }}</p>
              </div>
            </div>
          </BadgeRibbon>
        </template>
        <template v-else>
          <div v-for="historyItem in historyLoaded?.history" :key="historyItem.name">
            <div class="newHistoryItem">
              <p>{{ convertSymptom(historyItem.name.replace(/\s+/g, "") as SymptomKey) }}</p>
              <Switch v-model:checked="historyItem.value">
                <template #checkedChildren>
                  <CheckOutlined />
                </template>
                <template #unCheckedChildren>
                  <CloseOutlined />
                </template>
              </Switch>
            </div>
            <Divider style="height: 2px; background-color: #999" />
          </div>
          <Button type="primary" @click="evalHistory(historyLoaded?.history)">
            Eval all
          </Button>
          <Modal v-model:open="modalOpen" title="Basic Modal" @ok="handleOk">
            <p>Input Your Evaluation Title</p>
            <Input v-model:value="evaluationName" />
            <template #footer>
              <a-button key="submit" type="primary" @click="handleOk">
                Submit
              </a-button>
            </template>
          </Modal>
        </template>
      </Drawer>
    </div>
  </div>
</template>

<style scoped lang="scss">
.header {
  min-height: 60px;
  width: 100%;
  background-color: #ddcdb7;
  flex: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  .logo {
    display: flex;
    align-items: center;
    &:hover {
      cursor: pointer;
    }
  }
}

.title {
  font-size: 40px;
  font-family: NerkoOne;
  font-weight: 900;
  text-shadow:
    1px -1px #000,
    -1px 1px #999,
    -8px 8px 5px #80808088;
  color: #3e3e3e;
  margin: 0;
}
.historyDrawer {
  margin-right: 20px;
}
.historyItem {
  p {
    font-size: 15px;
  }
  margin-bottom: 14px;
  display: flex;
  gap: 10px;
  align-items: center;
  border: 1px solid #fff;
  border-radius: 5px;
  padding: 0 20px;
  background: #ffffff20;
  box-shadow: 0px 0px 5px 0px #4a4a4a;
  &:hover {
    cursor: pointer;
    background: #ffffff40;
    box-shadow: 0px 0px 10px 0px #4a4a4a;
  }
}
.newHistoryItem {
  p {
    font-size: 15px;
  }
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
}
.drawerTitle {
  display: flex;
  justify-content: space-between;
  align-items: center;
  .switchContainer {
    display: flex;
    justify-content: end;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    .ant-btn {
      margin-left: 20px;
    }
  }
}
</style>
