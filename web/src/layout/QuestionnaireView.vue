<script setup lang="ts">
import { ArrowLeftOutlined, ArrowRightOutlined, CheckOutlined, CloseOutlined } from '@ant-design/icons-vue'
import { Button, Input, Modal } from 'ant-design-vue'
import { requestPrediction, requestSymptom } from '~/requestWrapper'
import type { SymptomKey } from '~/requestWrapper/utils'
import { convertSymptom } from '~/requestWrapper/utils'
import { usePredictionResultStore } from '~/store/predictionResult'
import { useSymptomStore } from '~/store/symptom'

interface Question {
  answer?: boolean
  name: string
  isAnswer: boolean
}
const symptomStore = useSymptomStore()
const predictionStore = usePredictionResultStore()
const router = useRouter()

const evaluationName = ref<string>('New evaluation 1')
const questions = ref<Question[]>([])
const open = ref<boolean>(false)

function showModal() {
  open.value = true
}

function handleOk(e: MouseEvent) {
  console.log(e)
  open.value = false
  submitAllQuestion()
}
function next() {
  if (symptomStore.symptom.length > symptomStore.index + 1){
    symptomStore.increment()
  }
}
function previous() {
  if (symptomStore.index > 0){
    symptomStore.decrement()
  }
}
async function answerQuestion(answer: boolean) {
  questions.value[symptomStore.index].answer = answer
  questions.value[symptomStore.index].isAnswer = true
  symptomStore.changeHistory({
    name: questions.value[symptomStore.index].name,
    value: answer,
  })
  if (symptomStore.symptom.length === symptomStore.index + 1){
    open.value = true
    return
  }
  next()
}

async function submitAllQuestion(){
  if (symptomStore.symptom.length === symptomStore.index + 1){
    const newHistory = questions.value.map((item) => {
      return {
        name: item.name,
        value: item.answer as boolean,
      }
    })
    showModal()
    symptomStore.addHistories(newHistory, evaluationName.value)
    // 回答了所有问题
    const predictionResult = await requestPrediction({
      features: newHistory,
      worker_choice: 'cnn',
    })
    predictionStore.setPredictionResults(predictionResult)
    symptomStore.initNewHistory()
    symptomStore.initIndex()
    router.push({ name: 'Result' })
  }
}

onMounted(() => {
  const mountFunc = async () => {
    questions.value = (symptomStore.symptom.map((symptom) => {
      return {
        name: symptom,
        answer: false,
        isAnswer: false,
      }
    }))
  }
  mountFunc()
})
</script>

<template>
  <div class="questionnaire">
    <div v-if="questions.length" class="question">
      <Button class="arrowLeft" @click="previous">
        <ArrowLeftOutlined style="font-size: 40px;" />
      </Button>
      <div class="questionContent">
        <p>{{ `${symptomStore.index + 1}. ${convertSymptom(questions[symptomStore.index].name.replace(/\s+/g, "") as SymptomKey)}` }}</p>
      </div>
      <Button class="arrowRight" @click="next">
        <ArrowRightOutlined style="font-size: 40px;" />
      </Button>
    </div>
    <div class="buttonBox">
      <Button class="closed" @click="answerQuestion(false)">
        <CloseOutlined style="font-size: 100px;" />
      </Button>
      <Button class="checked" @click="answerQuestion(true)">
        <CheckOutlined style="font-size: 100px;" />
      </Button>
      <Modal v-model:open="open" title="Basic Modal" @ok="handleOk">
        <p>Input Your Evaluation Title</p>
        <Input v-model:value="evaluationName" />
        <template #footer>
          <a-button key="submit" type="primary" @click="handleOk">
            Submit
          </a-button>
        </template>
      </Modal>
    </div>
  </div>
</template>

<style scoped lang="less">
.question {
  .arrowLeft {
    padding: 0;
    width: 50px;
    max-width: 50px;
    height: 100%;
    background: none;
    border: none;
    &:hover {
      border: none;
    }
    &:active {
      border: none;
    }
    &:focus-visible {
      box-shadow: none;
      outline: none;
    }
  }
  .arrowRight {
    padding: 0;
    width: 50px;
    max-width: 50px;
    height: 100%;
    background: none;
    border: none;
    &:hover {
      border: none;
    }
    &:active {
      border: none;
    }
    &:focus-visible {
      box-shadow: none;
      outline: none;
    }
  }
  .questionContent {
    user-select: none;
    overflow: auto;
    height: 100%;
    padding: 0 0px;
    flex: 1;
    align-items: start;
  }
  p {
    text-align: center;
    font-family: NerkoOne;
    font-size: 8vw;
    font-weight: 500;
    line-height: 1;
    text-shadow:
      1px -1px #fff,
      -1px 1px #999,
      -8px 8px 5px #80808088;
    color: #d3d3d3;
    margin: 10px 0 0 0;
  }
  display: flex;
  justify-content: center;
  align-items: center;
  flex: 1;
  height: 0;
}

.buttonBox {
  display: flex;
  justify-content: center;
  flex: 0;
  min-height: 200px;
  display: flex;
  justify-content: center;
  .checked {
    height: 100%;
    width: 50%;
    max-width: 300px;
    background: #f56c6c40;
    &:hover {
      border: none;
      background: #f56c6c60;
    }
    &:active {
      border: none;
      background: #f56c6c80;
    }
    &:focus-visible {
      box-shadow: none;
      outline: none;
    }
  }
  .closed {
    height: 100%;
    width: 50%;
    max-width: 300px;
    background: #67c23a40;
    &:hover {
      border: none;
      background: #67c23a60;
    }
    &:active {
      border: none;
      background: #67c23a80;
    }
    &:focus-visible {
      box-shadow: none;
      outline: none;
    }
  }
}

.questionnaire {
  display: flex;
  height: 100%;
  width: 100%;
  flex-direction: column;
}
</style>
