<script lang="ts" setup>
const props = defineProps<{ size: number }>()
</script>

<template>
  <div class="icon" :style="{ width: `${size}px`, height: `${size}px` }">
    <svg xmlns="http://www.w3.org/2000/svg" :width="props.size" :height="props.size" viewBox="0 0 200 200">
      <!-- 定义滤镜 -->
      <defs>
        <filter id="drop-shadow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="-12" dy="12" stdDeviation="10" flood-color="black" flood-opacity="0.5" />
        </filter>
      </defs>
      <!-- Heart Shape -->
      <path filter="url(#drop-shadow)" d="M100 140c-8.3-8.2-16.6-16.4-25-25-13.5-13.4-25-25.3-25-40 0-12.4 10.1-22.5 22.5-22.5 8.8 0 16.5 5 20.5 12.5 4-7.5 11.7-12.5 20.5-12.5C140 52.5 150 62.6 150 75c0 14.7-11.5 26.6-25 40-8.4 8.6-16.7 16.8-25 25z" fill="#ff6b6b" />

      <!-- Prediction Graph Line -->
      <polyline points="50,110 70,90 85,120 115,80 135,95 150,60" fill="none" stroke="#2d98da" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />

      <!-- Dots on Graph -->
      <circle cx="50" cy="110" r="4" fill="#2d98da" />
      <circle cx="70" cy="90" r="4" fill="#2d98da" />
      <circle cx="85" cy="120" r="4" fill="#2d98da" />
      <circle cx="115" cy="80" r="4" fill="#2d98da" />
      <circle cx="135" cy="95" r="4" fill="#2d98da" />
      <circle cx="150" cy="60" r="4" fill="#2d98da" />
    </svg>
  </div>
</template>

<style lang="scss" scoped>
.icon {
  svg {
    * {
      box-shadow:
        1px -1px #fff,
        -1px 1px #999,
        -8px 8px 5px #80808088;
    }
  }
}
</style>
