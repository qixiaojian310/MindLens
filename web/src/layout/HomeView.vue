<script setup lang="ts">
import { Button } from 'ant-design-vue'

const router = useRouter()

const svgUrl = new URL('../assets/favicon.svg', import.meta.url).href

function enterAssessment() {
  router.push({ name: 'Questionnaire' })
}
toggleDark(false)
</script>

<template>
  <div class="home">
    <div class="titleBox">
      <div class="bgContainer">
        <img :src="svgUrl">
      </div>
      <p class="title">
        MindLens
      </p>
    </div>
    <div class="buttonBox">
      <Button @click="enterAssessment">
        Enter assessment
      </Button>
    </div>
  </div>
</template>

<style scoped lang="less">
.bgContainer {
  display: flex;
  align-items: center;
  justify-content: center;
  img {
    height: 3em;
  }
}
.titleBox {
  display: flex;
  width: 100%;
  font-size: 150px;
  align-items: center;
  justify-content: center;
}
.home {
  position: relative;
  display: flex;
  justify-content: center;
  flex-direction: column;
  align-items: center;
  height: 100%;
  width: 100%;
}
.title {
  text-align: center;
  font-family: NerkoOne;
  font-weight: 900;
  text-shadow:
    1px -1px #fff,
    -1px 1px #999,
    -8px 8px 5px #80808088;
  color: #e6e6e6;
  margin: 0;
}
.buttonBox {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  bottom: 150px;
}
</style>
