import { createPinia } from 'pinia'
import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

import routes from './route/index'

import './styles/main.css'
import './styles/fonts.css'

const pinia = createPinia()
const app = createApp(App)
const router = createRouter({
  routes,
  history: createWebHistory(import.meta.env.BASE_URL),
})
app.use(router).use(pinia)
app.mount('#app')
