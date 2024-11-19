<script setup lang="ts">
import { Button } from 'ant-design-vue'
import { requestPrediction, requestSymptom } from '~/requestWrapper'
import { useSymptomStore } from '~/store/symptom'

const symptomStore = useSymptomStore()

interface Question {
  answer?: boolean
  name: string
  isAnswer: boolean
}

const questions = ref<Question[]>([])
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
  if (symptomStore.symptom.length === symptomStore.index + 1){
    const predictionResult = await requestPrediction({
      features: questions.value.map((item) => {
        return {
          name: item.name,
          value: item.answer as boolean,
        }
      }),
      worker_choice: 'cnn',
    })
    console.log(predictionResult)
  }
  next()
}

onMounted(() => {
  const mountFunc = async () => {
    const symptomRaw = await requestSymptom()
    symptomStore.setSymptom(symptomRaw)
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
      <p>{{ `${symptomStore.index + 1}. ${questions[symptomStore.index].name}` }}</p>
    </div>
    <div class="buttonBox">
      <Button @click="previous">
        Previous Question
      </Button>
      <Button @click="answerQuestion(true)">
        Yes
      </Button>
      <Button @click="answerQuestion(false)">
        No
      </Button>
      <Button @click="next">
        Next Question
      </Button>
    </div>
  </div>
</template>

<style scoped lang="less">
.question {
  p {
    text-align: center;
    font-family: NerkoOne;
    font-size: 100px;
    font-weight: 900;
    line-height: 1;
    text-shadow:
      1px -1px #fff,
      -1px 1px #999,
      -8px 8px 5px #80808088;
    color: #d3d3d3;
    margin: 0;
  }
  display: flex;
  justify-content: center;
  align-items: center;
  flex: 1;
}

.buttonBox {
  display: flex;
  justify-content: center;
  flex: 0;
  min-height: 200px;
}

.questionnaire {
  display: flex;
  height: 100%;
  width: 100%;
  flex-direction: column;
}
</style>
