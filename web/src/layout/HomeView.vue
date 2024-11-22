<script setup lang="ts">
import { Button } from 'ant-design-vue'

const router = useRouter()

function enterAssessment() {
  router.push({ name: 'Questionnaire' })
}
toggleDark(false)
</script>

<template>
  <div class="home">
    <div class="titleBox">
      <div class="bgContainer">
        <div class="iconBG" />
        <LOGO :size="300" style="position: absolute; z-index: 2;" />
      </div>
      <p class="title">
        MindLens
      </p>
    </div>
    <div class="buttonBox">
      <Button @click="enterAssessment">
        Get your assessment
      </Button>
    </div>
  </div>
</template>

<style scoped lang="less">
.bgContainer {
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  height: 300px;
  width: 300px;
  .iconBG {
    position: absolute;
    z-index: 1;
    width: 170px;
    height: 170px;
    background-image: linear-gradient(15deg, #ff6b6b 85%, #0e72ca 15%);
    filter: blur(56px);
  }
}
.titleBox {
  padding-left: 56px;
  display: flex;
  width: 100%;
  align-items: center;
  gap: 100px;
  font-size: 150px;
  margin-bottom: 40px;
  justify-content: center;
  @media (max-width: 1000px) {
    flex-direction: column;
    gap: 0px;
    margin-bottom: 0;
    padding-left: 0px;
    font-size: 18vw;
  }
}
.home {
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
  .ant-btn {
    font-family: NerkoOne;
    height: 100px;
    color: #fff;
    font-size: 40px;
    background: #ff6b6b;
    box-shadow: 0 0 30px 10px #ff6b6b;
    border-radius: 10px;
    animation: breathing 2s infinite; /* 添加动画 */
    @media (max-width: 500px) {
      font-size: 6vw;
    }
  }
}
@keyframes breathing {
  0%,
  100% {
    box-shadow: 0 0 30px 10px rgba(255, 107, 107, 0.8); /* 初始状态 */
  }
  50% {
    box-shadow: 0 0 50px 20px rgba(255, 107, 107, 0.4); /* 呼吸扩展状态 */
  }
}
</style>
